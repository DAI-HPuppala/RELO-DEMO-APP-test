#!/usr/bin/env python3
"""
Test script to verify manual mode agent completion tracking.
This tests the new feature that tracks which agents have been completed
at least once and enables node clicking only after all three main agents
(initial_classifier, detail_extractor, damage_detector) have been completed.
"""

import asyncio
import logging
from backend.src.services.manual_mode_handler import ManualModeHandler
from backend.src.models.manual_session import ManualSessionState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_manual_mode_tracking():
    """Test the manual mode agent completion tracking"""
    
    # Create handler and session
    session_id = "test-session-123"
    handler = ManualModeHandler(session_id)
    session_state = ManualSessionState(session_id=session_id)
    
    print("\n=== Testing Manual Mode Agent Completion Tracking ===\n")
    
    # Activate manual mode
    handler.activate_manual_mode()
    
    # Set up a mock callback to capture status updates
    status_updates = []
    async def mock_send_status(status):
        status_updates.append(status)
        print(f"Status Update: node_clicking_enabled={status.get('node_clicking_enabled')}, "
              f"completed_set={status.get('completed_agents_set', [])}")
    
    handler.set_callbacks(send_status=mock_send_status)
    
    print("\n1. Initial state - no agents completed:")
    await handler._send_status_update()
    assert len(handler.completed_agents_set) == 0
    assert status_updates[-1]['node_clicking_enabled'] == False
    print(f"   ✓ Node clicking disabled, completed set: {handler.completed_agents_set}")
    
    print("\n2. Complete initial_classifier:")
    handler.mark_agent_completed('initial_classifier', {'type': 'shirt'})
    session_state.record_agent_completion('initial_classifier', {'type': 'shirt'})
    await handler._send_status_update()
    assert len(handler.completed_agents_set) == 1
    assert status_updates[-1]['node_clicking_enabled'] == False
    print(f"   ✓ Node clicking still disabled, completed set: {handler.completed_agents_set}")
    
    print("\n3. Complete detail_extractor:")
    handler.mark_agent_completed('detail_extractor', {'color': 'blue'})
    session_state.record_agent_completion('detail_extractor', {'color': 'blue'})
    await handler._send_status_update()
    assert len(handler.completed_agents_set) == 2
    assert status_updates[-1]['node_clicking_enabled'] == False
    print(f"   ✓ Node clicking still disabled, completed set: {handler.completed_agents_set}")
    
    print("\n4. Complete damage_detector (all 3 agents now complete):")
    handler.mark_agent_completed('damage_detector', {'damage': False})
    session_state.record_agent_completion('damage_detector', {'damage': False})
    await handler._send_status_update()
    assert len(handler.completed_agents_set) == 3
    assert status_updates[-1]['node_clicking_enabled'] == True
    print(f"   ✓ Node clicking ENABLED! Completed set: {handler.completed_agents_set}")
    
    print("\n5. Reset after final compiler (new cycle):")
    handler.reset_after_final_compiler()
    session_state.start_new_cycle()
    await handler._send_status_update()
    # The set should persist across cycles
    assert len(handler.completed_agents_set) == 3
    assert status_updates[-1]['node_clicking_enabled'] == True
    print(f"   ✓ Node clicking still ENABLED after new cycle! Completed set: {handler.completed_agents_set}")
    print(f"   ✓ Current cycle agents: {list(handler.completed_agents.keys())}")
    
    print("\n6. Complete initial_classifier again in new cycle:")
    handler.mark_agent_completed('initial_classifier', {'type': 'pants'})
    await handler._send_status_update()
    assert len(handler.completed_agents_set) == 3  # Still 3, not 4
    assert status_updates[-1]['node_clicking_enabled'] == True
    print(f"   ✓ Node clicking remains ENABLED, completed set unchanged: {handler.completed_agents_set}")
    
    print("\n7. Full reset (new session):")
    handler.reset()  # This should clear the set
    await handler._send_status_update()
    assert len(handler.completed_agents_set) == 0
    assert status_updates[-1]['node_clicking_enabled'] == False
    print(f"   ✓ Node clicking disabled after full reset, completed set cleared: {handler.completed_agents_set}")
    
    print("\n=== Test Passed! ===")
    print(f"\nSummary:")
    print(f"- Node clicking is disabled until all 3 agents complete at least once")
    print(f"- The completed_agents_set persists across cycles within a session")
    print(f"- The set is only cleared when starting a new session")
    print(f"- Total status updates sent: {len(status_updates)}")

if __name__ == "__main__":
    asyncio.run(test_manual_mode_tracking())