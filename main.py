import os
# Disable Tokenizers parallelism to avoid Rust panic in multi-threaded environments
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
from langchain_core.messages import HumanMessage

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.agents.graph import execute

def main():
    print("=== 南大SE流程智能体 (CLI) ===")
    print("上下文记忆功能已启用（会话内有效）。")
    print("输入 'q', 'quit', or 'exit' 退出程序。\n")

    # Define a static thread_id for local CLI session
    config = {"configurable": {"thread_id": "cli-user-1"}}

    while True:
        try:
            user_input = input("User: ").strip()
            if not user_input:
                continue
            
            if user_input.lower() in ["q", "quit", "exit"]:
                print("Bye!")
                break

            print("\nThinking...", end="", flush=True)
            
            # Initialize state
            # Add the current input as a HumanMessage to the messages list to maintain history
            initial_state = {
                "query": user_input,
                "messages": [HumanMessage(content=user_input)]
            }
            
            # Execute the graph with session config
            final_state = execute(initial_state, config=config)
            
            # Extract and print the answer
            final_answer = final_state.get("final_answer", "No answer generated.")
            print(f"\rAgent: {final_answer}\n")
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nBye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
