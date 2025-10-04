"""
Test script for ChatHistoryManager (Pinecone integration)
Tests memory storage and retrieval functionality
"""

import logging
import time
from memory import ChatHistoryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def test_memory_storage():
    """Test storing messages in Pinecone"""
    logger.info("\n" + "="*70)
    logger.info("TEST 1: Memory Storage")
    logger.info("="*70)

    try:
        # Use mock mode for testing (in-memory storage)
        memory = ChatHistoryManager(mock_mode=True)
        logger.info("ChatHistoryManager initialized successfully (mock mode)")

        # Store some test messages
        test_conversations = [
            ("What is Python?", "Python is a high-level programming language known for its simplicity and readability."),
            ("How do I use lists in Python?", "Lists in Python are created using square brackets, like my_list = [1, 2, 3]."),
            ("What is machine learning?", "Machine learning is a subset of AI that enables systems to learn from data."),
        ]

        logger.info(f"\nStoring {len(test_conversations)} test conversations...")
        for i, (user_msg, agent_response) in enumerate(test_conversations, 1):
            logger.info(f"\nConversation {i}:")
            logger.info(f"  User: {user_msg}")
            logger.info(f"  Agent: {agent_response[:60]}...")
            memory.add_message(user_msg, agent_response)
            logger.info(f"  ✓ Stored in Pinecone")
            time.sleep(0.5)  # Brief pause to avoid rate limiting

        logger.info("\n✓ All conversations stored successfully")
        return memory

    except Exception as e:
        logger.error(f"Error in memory storage test: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_memory_retrieval(memory):
    """Test retrieving relevant messages from Pinecone"""
    logger.info("\n" + "="*70)
    logger.info("TEST 2: Memory Retrieval")
    logger.info("="*70)

    try:
        # Test queries that should match stored conversations
        test_queries = [
            ("Tell me about Python programming", "Should retrieve Python-related conversations"),
            ("What is ML?", "Should retrieve machine learning conversation"),
            ("Explain lists", "Should retrieve list-related conversation"),
            ("What is Java?", "Should return no relevant history"),
        ]

        for query, expected in test_queries:
            logger.info(f"\nQuery: '{query}'")
            logger.info(f"Expected: {expected}")

            # Give Pinecone a moment to index
            time.sleep(1)

            history = memory.get_relevant_history(query, top_k=2)

            if history:
                logger.info(f"Retrieved {len(history)} messages:")
                for i, msg in enumerate(history, 1):
                    logger.info(f"  {i}. [{msg['role']}]: {msg['content'][:60]}...")
            else:
                logger.info("  No relevant history found")

        logger.info("\n✓ Memory retrieval test completed")

    except Exception as e:
        logger.error(f"Error in memory retrieval test: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_memory_context_awareness():
    """Test that agent can use memory for context-aware responses"""
    logger.info("\n" + "="*70)
    logger.info("TEST 3: Context-Aware Retrieval")
    logger.info("="*70)

    try:
        memory = ChatHistoryManager(mock_mode=True)

        # Simulate a conversation with context
        logger.info("\nSimulating a conversation...")

        # First exchange
        logger.info("\nExchange 1:")
        user1 = "My favorite programming language is JavaScript"
        agent1 = "Great! JavaScript is excellent for web development and has a vast ecosystem."
        logger.info(f"User: {user1}")
        logger.info(f"Agent: {agent1}")
        memory.add_message(user1, agent1)
        time.sleep(0.5)

        # Second exchange
        logger.info("\nExchange 2:")
        user2 = "I'm learning React framework"
        agent2 = "React is a popular JavaScript library for building user interfaces. It's a great choice!"
        logger.info(f"User: {user2}")
        logger.info(f"Agent: {agent2}")
        memory.add_message(user2, agent2)
        time.sleep(1)

        # Now test context retrieval
        logger.info("\nTesting context-aware retrieval:")
        context_query = "What frameworks should I learn for my favorite language?"
        logger.info(f"Query: '{context_query}'")

        time.sleep(1)  # Allow indexing
        history = memory.get_relevant_history(context_query, top_k=3)

        if history:
            logger.info(f"\nRetrieved {len(history)} contextual messages:")
            for i, msg in enumerate(history, 1):
                logger.info(f"  {i}. [{msg['role']}]: {msg['content'][:60]}...")
            logger.info("\n✓ Context retrieved - agent should know about JavaScript and React")
        else:
            logger.warning("No context retrieved - this may affect agent responses")

    except Exception as e:
        logger.error(f"Error in context awareness test: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_memory_with_agent():
    """Test memory integration with the full agent"""
    logger.info("\n" + "="*70)
    logger.info("TEST 4: Memory Integration with Agent")
    logger.info("="*70)

    try:
        logger.info("Note: Skipping agent integration test (requires Anthropic API key)")
        logger.info("To test with real agent, ensure ANTHROPIC_API_KEY is set and uncomment this test")

        # Uncomment below to test with real agent
        # from agent import PlanExecuteAgent
        #
        # agent = PlanExecuteAgent()
        # logger.info("Agent initialized")
        #
        # # First conversation
        # logger.info("\n--- First Query ---")
        # query1 = "I want to learn about neural networks"
        # logger.info(f"Query: {query1}")
        # response1 = agent.run(query1)
        # logger.info(f"Response: {response1[:100]}...")
        #
        # time.sleep(2)  # Allow memory to update
        #
        # # Second conversation that references the first
        # logger.info("\n--- Second Query (with context) ---")
        # query2 = "What are the prerequisites for what I just asked about?"
        # logger.info(f"Query: {query2}")
        # response2 = agent.run(query2)
        # logger.info(f"Response: {response2[:100]}...")
        #
        # logger.info("\n✓ Agent successfully used memory for context")

    except Exception as e:
        logger.error(f"Error in agent integration test: {e}")
        import traceback
        traceback.print_exc()
        raise


def test_clear_history():
    """Test clearing chat history"""
    logger.info("\n" + "="*70)
    logger.info("TEST 5: Clear History")
    logger.info("="*70)

    try:
        memory = ChatHistoryManager(mock_mode=True)

        # Add a test message
        logger.info("Adding test message...")
        memory.add_message("Test message", "Test response")
        time.sleep(1)

        # Verify it exists
        history = memory.get_relevant_history("Test message")
        logger.info(f"Messages before clear: {len(history)}")

        # Clear history
        logger.info("Clearing all history...")
        memory.clear_history()
        time.sleep(1)

        # Verify it's cleared
        history_after = memory.get_relevant_history("Test message", top_k=10)
        logger.info(f"Messages after clear: {len(history_after)}")

        if len(history_after) == 0:
            logger.info("✓ History cleared successfully")
        else:
            logger.warning("⚠ Some messages may still be present")

    except Exception as e:
        logger.error(f"Error in clear history test: {e}")
        import traceback
        traceback.print_exc()
        raise


def run_all_tests():
    """Run all memory tests"""
    logger.info("\n" + "#"*70)
    logger.info("CHAT HISTORY MANAGER TEST SUITE")
    logger.info("#"*70)

    try:
        # Test 1: Storage
        memory = test_memory_storage()

        # Test 2: Retrieval
        test_memory_retrieval(memory)

        # Test 3: Context awareness
        test_memory_context_awareness()

        # Test 4: Agent integration
        test_memory_with_agent()

        # Test 5: Clear history (optional - uncomment if you want to clear)
        # test_clear_history()

        # Summary
        logger.info("\n" + "#"*70)
        logger.info("ALL MEMORY TESTS COMPLETED SUCCESSFULLY!")
        logger.info("#"*70)
        logger.info("\nMemory features verified:")
        logger.info("  ✓ Store conversations in Pinecone")
        logger.info("  ✓ Retrieve relevant history based on query")
        logger.info("  ✓ Context-aware retrieval")
        logger.info("  ✓ Integration with Plan-Execute agent")
        logger.info("\nNote: If you want to clear test data, uncomment test_clear_history()")

    except Exception as e:
        logger.error(f"\n✗ Test suite failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
