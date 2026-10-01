import os
from dotenv import load_dotenv

# Load .env file
load_dotenv(override=True)

# Make sure ADK/Google GenAI can find the key
if os.getenv("GEMINI_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

import asyncio

from google.adk.runners import InMemoryRunner
from google.genai import types

from misinformation_agent.agent import root_agent


async def main():

    runner = InMemoryRunner(
        agent=root_agent,
        app_name="multimodal_misinformation"
    )

    user_id = "test_user"
    session_id = "test_session"

    # Create a session
    session = await runner.session_service.create_session(
        app_name="multimodal_misinformation",
        user_id=user_id,
        session_id=session_id
    )

    print("ADK session created.")
    print("Session ID:", session.id)

    # Test request
    message = types.Content(
    role="user",
    parts=[
        types.Part(
            text="""
Analyze this multimodal post for misinformation.

Text:
retired droids

Image path:
../objective2_ooc/test_images/test.jpg

Use the available analysis tools.
The image at the given path must be analyzed along with the text.
"""
        )
    ]
)


    print("\nSending request to Root Agent...\n")

    async for event in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=message
    ):

        if event.content and event.content.parts:

            for part in event.content.parts:

                if part.text:
                    print(part.text)


if __name__ == "__main__":
    asyncio.run(main())