"""Integration test for V2 stateful orchestration with multi-image inference"""

import asyncio
import pytest
import numpy as np
from unittest.mock import Mock, AsyncMock
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.v2.services.stateful_orchestrator import StatefulOrchestrator
from src.v2.services.session_state_manager import SessionStateManager
from src.v2.agents.initial_classifier_v2 import InitialClassifierV2
from src.v2.agents.detail_extractor_v2 import DetailExtractorV2
from src.v2.agents.damage_detector_v2 import DamageDetectorV2
from src.v2.agents.final_compiler_v2 import FinalCompilerV2


@pytest.mark.asyncio
async def test_full_orchestration_flow():
    """Test complete orchestration flow with all agents"""
    
    # Create orchestrator
    session_id = "test_session_123"
    orchestrator = StatefulOrchestrator(session_id)
    
    # Mock frame provider
    frame_count = 0
    async def mock_frame_provider():
        nonlocal frame_count
        frame_count += 1
        # Return a mock frame (640x480x3 RGB image)
        return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    orchestrator.frame_provider = mock_frame_provider
    
    # Mock message sender
    messages_sent = []
    async def mock_send_message(msg):
        messages_sent.append(msg)
    
    orchestrator.send_message = mock_send_message
    
    # Test custom timers (fast for testing)
    orchestrator.set_custom_timers({
        "initial_classifier": 0.5,  # 500ms
        "detail_extractor": 0.3,    # 300ms
        "damage_detector": 0.5      # 500ms
    })
    
    # Run initial classifier
    print("\n=== Testing Initial Classifier ===")
    agent_state = await orchestrator.run_agent_with_timer("initial_classifier")
    
    assert agent_state.agent_name == "initial_classifier"
    assert agent_state.status.value == "COMPLETED"
    assert agent_state.inference_count >= 1
    print(f"Initial classifier completed {agent_state.inference_count} inferences")
    
    # Check that first frame was shared
    shared_frame = await orchestrator.frame_registry.get_frame_for_damage_detector()
    assert shared_frame is not None, "Initial classifier should share first frame"
    
    # Run detail extractor
    print("\n=== Testing Detail Extractor ===")
    agent_state = await orchestrator.run_agent_with_timer("detail_extractor", 0.3)
    
    assert agent_state.agent_name == "detail_extractor"
    assert agent_state.status.value == "COMPLETED"
    print(f"Detail extractor completed {agent_state.inference_count} inferences")
    
    # Run damage detector
    print("\n=== Testing Damage Detector ===")
    agent_state = await orchestrator.run_agent_with_timer("damage_detector")
    
    assert agent_state.agent_name == "damage_detector"
    assert agent_state.status.value == "COMPLETED"
    assert agent_state.inference_count >= 1
    print(f"Damage detector completed {agent_state.inference_count} inferences")
    
    # Check session state
    assert orchestrator.session_state.agents_completed == ["initial_classifier", "detail_extractor", "damage_detector"]
    
    # Test pause/resume
    print("\n=== Testing Pause/Resume ===")
    
    # Start a new agent
    task = asyncio.create_task(orchestrator.run_agent_with_timer("initial_classifier", 2.0))
    
    # Wait a bit then pause
    await asyncio.sleep(0.1)
    pause_response = await orchestrator.pause_flow()
    
    assert pause_response["type"] == "flow_paused"
    assert orchestrator.session_state.status.value == "PAUSED"
    
    # Resume with restart
    resume_response = await orchestrator.resume_flow(restart_agent=True)
    
    assert resume_response["type"] == "flow_resumed"
    assert resume_response["restarted"] == True
    
    # Cancel the original task
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    
    # Test checkpoint persistence
    print("\n=== Testing Checkpoint Persistence ===")
    
    # Save checkpoint
    await orchestrator.state_persistence.save_checkpoint(orchestrator.session_state)
    
    # Get checkpoint info
    checkpoint_info = orchestrator.get_checkpoint_info()
    assert checkpoint_info["exists"] == True
    assert checkpoint_info["session_id"] == session_id
    
    # Create new orchestrator and load checkpoint
    new_orchestrator = StatefulOrchestrator(session_id)
    loaded = await new_orchestrator.load_from_checkpoint()
    
    assert loaded == True
    assert new_orchestrator.session_state.session_id == session_id
    assert new_orchestrator.session_state.agents_completed == ["initial_classifier", "detail_extractor", "damage_detector"]
    
    # Cleanup
    await orchestrator.cleanup()
    await new_orchestrator.cleanup()
    
    print("\n=== All Tests Passed! ===")
    print(f"Total frames captured: {frame_count}")
    print(f"Total messages sent: {len(messages_sent)}")


@pytest.mark.asyncio
async def test_multi_image_inference():
    """Test multi-image inference with progressive frame accumulation"""
    
    # Create initial classifier
    classifier = InitialClassifierV2()
    
    # Mock frame provider
    frames_provided = []
    async def mock_frame_provider():
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        frames_provided.append(frame)
        return frame
    
    classifier.set_frame_provider(mock_frame_provider)
    
    # Set fast timer for testing
    classifier.timer_seconds = 1.0
    
    # Run with timer
    result = await classifier.run_with_timer()
    
    # Check results
    assert "attributes" in result
    assert "total_inferences" in result
    assert result["total_inferences"] >= 1
    
    print(f"\nClassifier completed {result['total_inferences']} inferences")
    print(f"Total frames captured: {len(frames_provided)}")
    print(f"Aggregation method: {result.get('aggregation_method', 'N/A')}")
    
    # For multiple inferences, check aggregation
    if result["total_inferences"] > 1:
        assert result["aggregation_method"] in ["single", "last_with_fallback", "majority_voting"]


@pytest.mark.asyncio
async def test_frame_sharing_protocol():
    """Test frame sharing between initial classifier and damage detector"""
    
    session_id = "test_frame_sharing"
    orchestrator = StatefulOrchestrator(session_id)
    
    # Mock frame provider
    async def mock_frame_provider():
        return np.ones((480, 640, 3), dtype=np.uint8) * 100  # Gray frame
    
    orchestrator.frame_provider = mock_frame_provider
    
    # Create agents
    initial = InitialClassifierV2()
    damage = DamageDetectorV2()
    
    # Wire them up
    initial.set_frame_registry(orchestrator.frame_registry)
    initial.set_frame_provider(mock_frame_provider)
    
    damage.set_frame_registry(orchestrator.frame_registry)
    damage.set_frame_provider(mock_frame_provider)
    
    # Set fast timers
    initial.timer_seconds = 0.3
    damage.timer_seconds = 0.3
    
    # Run initial classifier
    await initial.run_with_timer()
    
    # Check frame was shared
    shared_frame = await orchestrator.frame_registry.get_frame_for_damage_detector()
    assert shared_frame is not None
    assert shared_frame.shape == (480, 640, 3)
    assert np.all(shared_frame == 100)  # Should be the gray frame
    
    # Run damage detector
    result = await damage.run_with_timer()
    
    # Damage detector's first inference should have used 2 frames (shared + new)
    assert result["total_inferences"] >= 1
    
    # Cleanup
    await orchestrator.cleanup()
    
    print("\nFrame sharing protocol test passed!")


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_full_orchestration_flow())
    asyncio.run(test_multi_image_inference())
    asyncio.run(test_frame_sharing_protocol())
    
    print("\n✅ All integration tests passed!")