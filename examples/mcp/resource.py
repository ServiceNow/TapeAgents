import json
import os

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("File access resource")
root_dir = "/tmp/file_access_demo/"


@mcp.resource("data://workspace")
def get_workspace_files() -> str:
    """Get the list of files in the workspace directory"""
    os.makedirs(root_dir, exist_ok=True)
    files = os.listdir(root_dir)
    content = json.dumps(files)
    return content


@mcp.resource("data://workspace/{file_name}")
def get_file_content(file_name: str) -> str:
    """Get the content of a file in the workspace directory"""
    file_path = os.path.join(root_dir, file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_name} not found in workspace")
    with open(file_path, "r") as f:
        content = f.read()
    return content


@mcp.resource("config://demo")
def get_config() -> str:
    """Static configuration data"""
    config = {
        "name": "File Access App",
        "version": "1.0",
        "description": "This is a demo config resource",
    }
    return json.dumps(config)


if __name__ == "__main__":
    mcp.run()
