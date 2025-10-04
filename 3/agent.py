"""
Simple ReAct Agent using LangGraph

This agent follows the ReAct (Reasoning + Acting) pattern:
1. Thinks about what to do next (Reasoning)
2. Takes an action (Acting) - uses tools or provides final answer
3. Observes the result
4. Repeats until the task is complete
"""

import os
import logging
from typing import TypedDict, List, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

from tools.tools import get_tools
from memory import ChatHistoryManager

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Define the state
class AgentState(TypedDict):
    """State of the ReAct agent"""
    input: str
    chat_history: List[dict]
    messages: List
    iterations: int
    final_answer: str


class ReActAgent:
    """ReAct Agent implementation with tool calling"""

    def __init__(self, verbose: bool = True, max_iterations: int = 5):
        self.verbose = verbose
        self.max_iterations = max_iterations

        # Get tools
        self.tools = get_tools()

        # Initialize LLM with tool binding
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=0
        ).bind_tools(self.tools)

        self.memory = ChatHistoryManager()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow for ReAct"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", self.tools_node)

        # Add edges
        workflow.set_entry_point("agent")
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "continue": "tools",
                "end": END
            }
        )
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    def agent_node(self, state: AgentState) -> AgentState:
        """Agent reasoning step: Decide what to do next"""
        # Retrieve chat history if first iteration
        if state["iterations"] == 0:
            history = self.memory.get_relevant_history(state["input"])
            state["chat_history"] = history

            if history:
                logger.info(f"💭 Using {len(history)} messages from chat history")
                # Add history to messages
                for msg in history[-4:]:
                    if msg["role"] == "user":
                        state["messages"].append(HumanMessage(content=msg["content"]))
                    else:
                        state["messages"].append(AIMessage(content=msg["content"]))

            # Add the current user input
            state["messages"].append(HumanMessage(content=state["input"]))

        response = self.llm.invoke(state["messages"])

        # Log the response
        if response.tool_calls:
            logger.info(f"🔧 Using tool: {response.tool_calls[0]['name']} - {response.tool_calls[0]['args']['query'] if 'query' in response.tool_calls[0]['args'] else response.tool_calls[0]['args']}")

        # Add response to messages
        state["messages"].append(response)
        state["iterations"] += 1

        return state

    def tools_node(self, state: AgentState) -> AgentState:
        """Execute tools based on agent's decision"""
        # Get the last message (should be AIMessage with tool calls)
        last_message = state["messages"][-1]

        if not last_message.tool_calls:
            return state

        # Execute each tool call
        tool_messages = []
        for tool_call in last_message.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # Find and execute the tool
            tool = next((t for t in self.tools if t.name == tool_name), None)
            if tool:
                result = tool.invoke(tool_args)
                tool_messages.append(
                    ToolMessage(
                        content=result,
                        tool_call_id=tool_call["id"]
                    )
                )
            else:
                logger.error(f"❌ Tool {tool_name} not found")
                tool_messages.append(
                    ToolMessage(
                        content=f"Error: Tool {tool_name} not found",
                        tool_call_id=tool_call["id"]
                    )
                )

        # Add tool messages to state
        state["messages"].extend(tool_messages)

        return state

    def should_continue(self, state: AgentState) -> Literal["continue", "end"]:
        """Decide whether to continue or end"""
        if not state["messages"]:
            return "end"

        last_message = state["messages"][-1]

        # If the last message has tool calls, continue to tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "continue"

        # If no tool calls, we have a final answer
        if isinstance(last_message, AIMessage) and last_message.content:
            return "end"

        # Check iteration limit
        if state["iterations"] >= self.max_iterations:
            logger.warning(f"⚠️  Max iterations ({self.max_iterations}) reached")
            return "end"

        return "continue"

    def run(self, user_input: str) -> str:
        """Run the agent with a user input"""
        initial_state = {
            "input": user_input,
            "chat_history": [],
            "messages": [],
            "iterations": 0,
            "final_answer": ""
        }

        result = self.graph.invoke(initial_state)

        # Extract final answer from the last AI message
        final_answer = ""
        for message in reversed(result["messages"]):
            if isinstance(message, AIMessage) and message.content:
                final_answer = message.content
                break

        # Store in chat history
        if final_answer:
            self.memory.add_message(user_input, final_answer)

        return final_answer


def main():
    """Main function to run the agent interactively"""
    agent = ReActAgent()

    print("ReAct Agent (type 'quit' to exit)")
    print("=" * 50)

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break

        if not user_input:
            continue

        try:
            response = agent.run(user_input)
            print(f"\nAgent: {response}")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
