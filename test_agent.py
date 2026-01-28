import asyncio
import json
import os

from openai import AsyncOpenAI
from openreward import AsyncOpenReward

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-5.2")
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]


async def main() -> None:
    or_client = AsyncOpenReward()
    oai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    environment = or_client.environments.get(name="local/Whodunnit", base_url="http://localhost:8080")
    tasks = await environment.list_tasks(split="test")
    tools = await environment.list_tools(format="openai")

    print(f"Found {len(tasks)} tasks")

    # Test first task
    task = tasks[4]
    print(task)

    async with environment.session(task=task) as session:
        prompt = await session.get_prompt()

        # Handle both string and TextBlock responses
        if isinstance(prompt, list):
            content = prompt[0].text
        else:
            content = prompt

        input_list = [{"role": "user", "content": content}]
        finished = False
        print(input_list[-1])

        turn = 0
        max_turns = 20  # Safety limit

        while not finished and turn < max_turns:
            turn += 1
            print(f"\n--- Turn {turn} ---")

            response = await oai_client.responses.create(
                model=MODEL_NAME,
                tools=tools,
                input=input_list,
            )

            input_list += response.output
            print(input_list[-1])

            for item in response.output:
                if item.type == "function_call":
                    print(f"Tool called: {item.name}")

                    tool_result = await session.call_tool(
                        item.name,
                        json.loads(str(item.arguments)),
                    )

                    finished = tool_result.finished
                    reward = tool_result.reward

                    # Add tool result to input
                    result_text = (
                        tool_result.blocks[0].text if tool_result.blocks else ""
                    )
                    input_list.append(
                        {
                            "type": "function_call_output",
                            "call_id": item.call_id,
                            "output": result_text,
                        }
                    )
                    print(input_list[-1])

                    if finished:
                        print(f"\n{'='*50}")
                        print(f"FINAL REWARD: {reward:.0%}")
                        print(f"{'='*50}")
                        print(f"\n{result_text}")
                        break

                elif item.type == "message":
                    # Model's reasoning (if any)
                    if hasattr(item, "content") and item.content:
                        print(f"Model reasoning: {item.content[:200]}...")

            # Break if no tool calls (model gave up)
            if not any(i.type == "function_call" for i in response.output):
                print("Model did not call any tools, stopping.")
                break

        if turn >= max_turns:
            print(f"\nReached max turns ({max_turns}) without finishing.")


if __name__ == "__main__":
    asyncio.run(main())
