from langchain_core.tools import tool

@tool
def multiply(a: int, b: int) -> int:
    """Multiplies two integer and returns the integer result."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """Adds two integer and returns the integer result."""
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtracts two integer and returns the integer result."""
    return a - b



class MathToolkit:
    """A toolkit for basic math operations."""

    def get_tools(self):
        return [multiply, add, subtract]


math = MathToolkit()

math_tools = math.get_tools()

for tool in math_tools:
    print(tool.name, "=>" ,tool.description)


