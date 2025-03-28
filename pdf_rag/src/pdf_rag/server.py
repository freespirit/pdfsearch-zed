import asyncio

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from pdf_rag.env import load_env_file
from pdf_rag.rag import RAG

# Define available prompts
PROMPTS = {
    "pdfsearch": types.Prompt(
        name="pdfsearch",
        description="Do a RAG-style expansion of your prompt, enriching it with relevant information from the PDF.",
        arguments=[
            types.PromptArgument(
                name="input",
                description="What to look for in the document.",
                required=True,
            )
        ],
    )
}

load_env_file()

# Initialize server
app = Server("pdfsearch-server")


@app.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return list(PROMPTS.values())


@app.get_prompt()
async def get_prompt(
    name: str, arguments: dict[str, str] | None = None
) -> types.GetPromptResult:
    if name not in PROMPTS:
        raise ValueError(f"Prompt not found: {name}")

    if name == "pdfsearch":
        user_input = arguments.get("input") if arguments else ""

        # TODO figure out when to build the vector db
        rag = RAG()
        related_chunks = await rag.search(user_input)
        response = ""
        for chunk in related_chunks:
            response += "<text>\n"
            response += chunk
            response += "<//text>\n"
        response += "\n"

        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text=response,
                    ),
                )
            ]
        )

    raise ValueError("Prompt implementation not found")


async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="pdf_rag",
                server_version="0.1.0",
                capabilities=app.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


# This is needed if you'd like to connect to a custom client
if __name__ == "__main__":
    asyncio.run(main())
