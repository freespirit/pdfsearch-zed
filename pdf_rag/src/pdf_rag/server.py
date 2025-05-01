import asyncio

import mcp.server.stdio
import mcp.types as types
from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions

from pdf_rag.env import load_env_file
from pdf_rag.rag import RAG

load_env_file()

server = Server("pdfsearch-server")


@server.list_prompts()
async def list_prompts() -> list[types.Prompt]:
    return [
        types.Prompt(
            name="pdfsearch",
            description="Do a RAG-style expansion of your prompt, enriching it with relevant information from the PDF.",
            arguments=[
                types.PromptArgument(
                    name="input",
                    description="What to look for in the document.",
                    required=True,
                )
            ]
        )
    ]


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="pdfsearch",
            description="Retrieve relevant information from a document.",
            inputSchema={
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "User provided query (to search for in the documents)",
                    }
                },
            },
        )
    ]


@server.get_prompt()
async def get_prompt(
        name: str,
        arguments: dict[str, str] | None = None
) -> types.GetPromptResult:
    if name != "pdfsearch":
        raise ValueError(f"Prompt not found: {name}")

    user_input = arguments.get("input") if arguments else ""
    result = await _search(user_input)
    return types.GetPromptResult(
        messages=[
            types.PromptMessage(
                role="user",
                content=types.TextContent(
                    type="text",
                    text=result,
                ),
            )
        ]
    )


@server.call_tool()
async def call_a_tool(
        name: str,
        arguments: dict
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    if name != "pdfsearch":
        raise ValueError(f"Unknown tool: {name}")
    # if "query" not in arguments:
    #     raise ValueError("Missing required argument 'query'")

    result = f"received: {arguments}"
    try:
        user_input = arguments["query"]
        result = await _search(user_input)
    except Exception as e:
        result = str(e.__repr__())

    return [types.TextContent(type="text", text=result)]


async def _search(user_input):
    # TODO figure out when to build the vector db
    rag = RAG()
    related_chunks = await rag.search(user_input)
    response = ""
    for chunk in related_chunks:
        response += "<text>\n"
        response += chunk
        response += "</text>\n"
    response += "\n"
    return response


async def main():
    # Run the server using stdin/stdout streams
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="pdf_rag",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )


# This is needed if you'd like to connect to a custom client
if __name__ == "__main__":
    asyncio.run(main())
