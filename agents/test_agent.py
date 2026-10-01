from misinformation_agent.agent import root_agent


print("ADK Root Agent loaded successfully!")

print("Agent name:", root_agent.name)

print("Available tools:")

for tool in root_agent.tools:
    print(" -", tool.name if hasattr(tool, "name") else tool.__name__)