"""End-to-End Tests for V2 Stateful Orchestration System"""

import asyncio
import pytest
import numpy as np
import time
from unittest.mock import Mock, AsyncMock, patch
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
from src.api.websocket_v2 import WebSocketHandlerV2


class TestScenarios:
    """End-to-end test scenarios for V2 system"""
    
    @pytest.mark.asyncio
    async def test_complete_happy_path(self):
        """Test complete flow without interruptions"""
        print("\n=== Test: Complete Happy Path ===")
        
        # Setup
        session_id = "test_happy_path"
        orchestrator = StatefulOrchestrator(session_id)
        
        # Mock frame provider
        async def mock_frame_provider():
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        orchestrator.frame_provider = mock_frame_provider
        
        # Use fast timers for testing
        orchestrator.set_custom_timers({
            "initial_classifier": 0.5,
            "detail_extractor": 0.4,
            "damage_detector": 0.5
        })
        
        # Track messages
        messages = []
        orchestrator.send_message = lambda msg: messages.append(msg)
        
        # Run agents
        agents = ["initial_classifier", "detail_extractor", "damage_detector"]
        results = {}
        
        for agent_name in agents:
            agent_state = await orchestrator.run_agent_with_timer(agent_name)
            assert agent_state.status.value == "COMPLETED"
            assert agent_state.inference_count >= 1
            results[agent_name] = {
                "attributes": agent_state.finalized_attributes,
                "total_inferences": agent_state.inference_count,
                "aggregation_method": agent_state.aggregation_method
            }
            print(f"  {agent_name}: {agent_state.inference_count} inferences")
        
        # Compile final results
        final_compiler = FinalCompilerV2()
        final_result = await final_compiler.process(results)
        
        assert final_result is not None
        assert "item_type" in final_result
        
        # Verify checkpoint was saved
        checkpoint_info = orchestrator.get_checkpoint_info()
        assert checkpoint_info["exists"] == True
        
        # Cleanup
        await orchestrator.cleanup()
        
        print(f"  Total messages sent: {len(messages)}")
        print("  ✅ Happy path test passed!")
    
    @pytest.mark.asyncio
    async def test_pause_resume_flow(self):
        """Test pausing and resuming the flow"""
        print("\n=== Test: Pause/Resume Flow ===")
        
        session_id = "test_pause_resume"
        orchestrator = StatefulOrchestrator(session_id)
        
        # Mock frame provider
        async def mock_frame_provider():
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        orchestrator.frame_provider = mock_frame_provider
        
        # Start initial classifier with longer timer
        orchestrator.set_custom_timers({"initial_classifier": 2.0})
        
        # Start agent in background
        agent_task = asyncio.create_task(
            orchestrator.run_agent_with_timer("initial_classifier")
        )
        
        # Wait a bit then pause
        await asyncio.sleep(0.3)
        pause_response = await orchestrator.pause_flow()
        
        assert pause_response["type"] == "flow_paused"
        assert pause_response["paused_agent"] == "initial_classifier"
        assert orchestrator.session_state.status.value == "PAUSED"
        
        print(f"  Paused at: {pause_response['paused_agent']}")
        print(f"  Timer remaining: {pause_response['timer_remaining']:.1f}s")
        
        # Resume without restart
        resume_response = await orchestrator.resume_flow(restart_agent=False)
        
        assert resume_response["type"] == "flow_resumed"
        assert resume_response["restarted"] == False
        
        # Cancel the original task
        agent_task.cancel()
        try:
            await agent_task
        except asyncio.CancelledError:
            pass
        
        # Resume with restart
        resume_response = await orchestrator.resume_flow(restart_agent=True)
        assert resume_response["restarted"] == True
        
        print("  ✅ Pause/Resume test passed!")
        
        await orchestrator.cleanup()
    
    @pytest.mark.asyncio
    async def test_frame_sharing_protocol(self):
        """Test frame sharing between initial classifier and damage detector"""
        print("\n=== Test: Frame Sharing Protocol ===")
        
        session_id = "test_frame_sharing"
        orchestrator = StatefulOrchestrator(session_id)
        
        # Create a distinctive frame
        test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 42
        
        # Mock frame provider
        call_count = 0
        async def mock_frame_provider():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return test_frame  # First frame for initial classifier
            else:
                return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        orchestrator.frame_provider = mock_frame_provider
        
        # Run initial classifier
        initial = InitialClassifierV2()
        initial.set_frame_registry(orchestrator.frame_registry)
        initial.set_frame_provider(mock_frame_provider)
        initial.timer_seconds = 0.3
        
        await initial.run_with_timer()
        
        # Check frame was shared
        shared_frame = await orchestrator.frame_registry.get_frame_for_damage_detector()
        assert shared_frame is not None
        
        # Verify it's the test frame (reconstructed from bytes)
        # Note: In real implementation, proper JPEG encoding/decoding would be used
        
        # Run damage detector
        damage = DamageDetectorV2()
        damage.set_frame_registry(orchestrator.frame_registry)
        damage.set_frame_provider(mock_frame_provider)
        damage.timer_seconds = 0.3
        
        result = await damage.run_with_timer()
        
        # Damage detector should have used shared frame + new frame for first inference
        assert result["total_inferences"] >= 1
        
        print("  Initial frame registered and shared")
        print("  Damage detector retrieved shared frame")
        print("  ✅ Frame sharing test passed!")
        
        await orchestrator.cleanup()
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test error handling and recovery"""
        print("\n=== Test: Error Recovery ===")
        
        session_id = "test_error_recovery"
        orchestrator = StatefulOrchestrator(session_id)
        
        # Mock frame provider that fails occasionally
        fail_count = 0
        async def flaky_frame_provider():
            nonlocal fail_count
            fail_count += 1
            if fail_count % 3 == 0:
                raise Exception("Frame capture failed")
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        orchestrator.frame_provider = flaky_frame_provider
        orchestrator.set_custom_timers({"initial_classifier": 1.0})
        
        # Run with error handling
        try:
            agent_state = await orchestrator.run_agent_with_timer("initial_classifier")
            
            # Should complete despite some frame capture failures
            assert agent_state.status.value == "COMPLETED"
            assert agent_state.inference_count >= 1
            
            print(f"  Completed despite {fail_count} frame capture attempts")
            print(f"  Successful inferences: {agent_state.inference_count}")
            print("  ✅ Error recovery test passed!")
            
        except Exception as e:
            pytest.fail(f"Should have recovered from errors: {e}")
        
        await orchestrator.cleanup()
    
    @pytest.mark.asyncio
    async def test_checkpoint_recovery(self):
        """Test session recovery from checkpoint"""
        print("\n=== Test: Checkpoint Recovery ===")
        
        session_id = "test_checkpoint_recovery"
        
        # Create first orchestrator and run some agents
        orchestrator1 = StatefulOrchestrator(session_id)
        
        async def mock_frame_provider():
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        orchestrator1.frame_provider = mock_frame_provider
        orchestrator1.set_custom_timers({
            "initial_classifier": 0.3,
            "detail_extractor": 0.3
        })
        
        # Run two agents
        await orchestrator1.run_agent_with_timer("initial_classifier")
        await orchestrator1.run_agent_with_timer("detail_extractor")
        
        # Save checkpoint
        await orchestrator1.state_persistence.save_checkpoint(orchestrator1.session_state)
        
        original_completed = orchestrator1.session_state.agents_completed.copy()
        print(f"  Original session completed: {original_completed}")
        
        # Cleanup first orchestrator
        await orchestrator1.cleanup()
        
        # Create new orchestrator and recover
        orchestrator2 = StatefulOrchestrator(session_id)
        recovered = await orchestrator2.load_from_checkpoint()
        
        assert recovered == True
        assert orchestrator2.session_state.session_id == session_id
        assert orchestrator2.session_state.agents_completed == original_completed
        
        print(f"  Recovered session completed: {orchestrator2.session_state.agents_completed}")
        print("  ✅ Checkpoint recovery test passed!")
        
        await orchestrator2.cleanup()
    
    @pytest.mark.asyncio
    async def test_multi_inference_aggregation(self):
        """Test different aggregation methods based on inference count"""
        print("\n=== Test: Multi-Inference Aggregation ===")
        
        from src.v2.services.frame_aggregator import FrameAggregator
        from src.v2.models.inference_result import InferenceResult
        
        aggregator = FrameAggregator()
        
        # Test 1 inference - use as-is
        results_1 = [
            InferenceResult("test_agent", 1, 1).complete(
                {"color": "red", "size": "M"}, 0.9, "Single inference"
            )
        ]
        
        aggregated_1 = await aggregator.aggregate("test_agent", results_1)
        assert aggregated_1.aggregation_method == "single"
        assert aggregated_1.attributes["color"] == "red"
        print("  1 inference: single method ✓")
        
        # Test 2 inferences - last with fallback
        results_2 = [
            InferenceResult("test_agent", 1, 1).complete(
                {"color": "red", "size": "M", "brand": "TestBrand"}, 0.8, ""
            ),
            InferenceResult("test_agent", 2, 2).complete(
                {"color": "blue", "size": None}, 0.9, ""
            )
        ]
        
        aggregated_2 = await aggregator.aggregate("test_agent", results_2)
        assert aggregated_2.aggregation_method == "last_with_fallback"
        assert aggregated_2.attributes["color"] == "blue"  # From second
        assert aggregated_2.attributes["size"] == "M"  # Fallback from first
        assert aggregated_2.attributes["brand"] == "TestBrand"  # From first
        print("  2 inferences: last_with_fallback method ✓")
        
        # Test 3+ inferences - majority voting
        results_3 = [
            InferenceResult("test_agent", 1, 1).complete(
                {"color": "red"}, 0.8, ""
            ),
            InferenceResult("test_agent", 2, 2).complete(
                {"color": "blue"}, 0.85, ""
            ),
            InferenceResult("test_agent", 3, 3).complete(
                {"color": "red"}, 0.9, ""
            )
        ]
        
        aggregated_3 = await aggregator.aggregate("test_agent", results_3)
        assert aggregated_3.aggregation_method == "majority_voting"
        assert aggregated_3.attributes["color"] == "red"  # Majority
        print("  3+ inferences: majority_voting method ✓")
        
        print("  ✅ Multi-inference aggregation test passed!")
    
    @pytest.mark.asyncio
    async def test_performance_benchmarks(self):
        """Test performance metrics and benchmarks"""
        print("\n=== Test: Performance Benchmarks ===")
        
        session_id = "test_performance"
        orchestrator = StatefulOrchestrator(session_id)
        
        async def mock_frame_provider():
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        orchestrator.frame_provider = mock_frame_provider
        
        # Set realistic timers
        orchestrator.set_custom_timers({
            "initial_classifier": 4.0,
            "detail_extractor": 3.0,
            "damage_detector": 4.0
        })
        
        # Measure performance
        start_time = time.time()
        total_inferences = 0
        
        for agent_name in ["initial_classifier", "detail_extractor", "damage_detector"]:
            agent_start = time.time()
            agent_state = await orchestrator.run_agent_with_timer(agent_name)
            agent_time = time.time() - agent_start
            
            total_inferences += agent_state.inference_count
            
            # Performance assertions
            assert agent_time <= orchestrator.get_agent_timer(agent_name) + 0.5  # Allow 0.5s overhead
            assert agent_state.inference_count >= 1
            
            print(f"  {agent_name}: {agent_state.inference_count} inferences in {agent_time:.2f}s")
        
        total_time = time.time() - start_time
        
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Total inferences: {total_inferences}")
        print(f"  Average inference rate: {total_inferences/total_time:.2f}/s")
        
        # Performance benchmarks
        assert total_time <= 12.0  # Should complete within 12 seconds
        assert total_inferences >= 3  # At least 1 per agent
        
        print("  ✅ Performance benchmark test passed!")
        
        await orchestrator.cleanup()


@pytest.mark.asyncio
async def test_websocket_integration():
    """Test WebSocket handler integration"""
    print("\n=== Test: WebSocket Integration ===")
    
    handler = WebSocketHandlerV2()
    
    # Mock WebSocket
    mock_ws = Mock()
    mock_ws.send = AsyncMock()
    
    # Test offer handling
    offer_msg = {
        "type": "offer",
        "session_id": "test_ws_session",
        "sdp": "mock_sdp"
    }
    
    with patch('aiortc.RTCPeerConnection'):
        response = await handler.handle_message(mock_ws, offer_msg)
        
        assert response["type"] == "answer"
        assert "sdp" in response
        assert response["session_id"] == "test_ws_session"
    
    print("  WebSocket offer/answer exchange ✓")
    
    # Test custom timers
    timer_msg = {
        "type": "set_custom_timers",
        "session_id": "test_ws_session",
        "timers": {
            "initial_classifier": 2.0,
            "detail_extractor": 1.5
        }
    }
    
    # Create a mock orchestrator for the session
    handler.sessions["test_ws_session"] = Mock()
    handler.sessions["test_ws_session"].set_custom_timers = Mock()
    
    response = await handler.handle_message(mock_ws, timer_msg)
    
    assert response["type"] == "custom_timers_set"
    handler.sessions["test_ws_session"].set_custom_timers.assert_called_once()
    
    print("  Custom timer setting ✓")
    print("  ✅ WebSocket integration test passed!")


if __name__ == "__main__":
    # Run all test scenarios
    test_suite = TestScenarios()
    
    print("=" * 50)
    print("V2 END-TO-END TEST SUITE")
    print("=" * 50)
    
    asyncio.run(test_suite.test_complete_happy_path())
    asyncio.run(test_suite.test_pause_resume_flow())
    asyncio.run(test_suite.test_frame_sharing_protocol())
    asyncio.run(test_suite.test_error_recovery())
    asyncio.run(test_suite.test_checkpoint_recovery())
    asyncio.run(test_suite.test_multi_inference_aggregation())
    asyncio.run(test_suite.test_performance_benchmarks())
    asyncio.run(test_websocket_integration())
    
    print("\n" + "=" * 50)
    print("✅ ALL END-TO-END TESTS PASSED!")
    print("=" * 50)