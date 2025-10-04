"""
Test script for the ReAct Agent
Run this to test the agent without interactive mode
"""

import logging
from agent import ReActAgent

# Configure logging to show all agent reasoning
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

def test_simple_query():
    """Test the agent with a simple query"""
    print("\n" + "="*70)
    print("TEST 1: Simple Math Query")
    print("="*70)

    agent = ReActAgent()
    query = "What is 15 + 27?"

    response = agent.run(query)
    print(f"\n>>> FINAL RESPONSE: {response}\n")


def test_reasoning_query():
    """Test the agent with a reasoning query"""
    print("\n" + "="*70)
    print("TEST 2: Multi-Step Reasoning")
    print("="*70)

    agent = ReActAgent()
    query = "If a train travels 60 miles per hour for 2.5 hours, how far does it travel?"

    response = agent.run(query)
    print(f"\n>>> FINAL RESPONSE: {response}\n")


def test_all():
    """Run all tests"""
    try:
        test_simple_query()
        test_reasoning_query()

        print("\n" + "="*70)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70 + "\n")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_all()
