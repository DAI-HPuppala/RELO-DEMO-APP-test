#!/usr/bin/env python3
"""Test script to verify agent prompts are working correctly"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend/src'))

from v2.services.multi_inference_engine import MultiInferenceEngine

# Create an instance
engine = MultiInferenceEngine()

# Test each agent's prompt
agents = ["initial_classifier", "detail_extractor", "damage_detector", "final_compiler"]

print("=" * 60)
print("AGENT PROMPTS TEST")
print("=" * 60)

for agent in agents:
    prompt = engine._get_agent_prompt(agent, 1)
    print(f"\n{agent}:")
    print(f"  Prompt: {prompt}")
    print()

print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)