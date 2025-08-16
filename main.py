import os
from dotenv import load_dotenv
from agents import Agent, Runner, OpenAIChatCompletionsModel, AsyncOpenAI, function_tool, set_tracing_disabled
from pydantic import BaseModel
from typing import Any

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

set_tracing_disabled(disabled=True)

provider = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

gemin_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)

inventory_data = [
    {"id": 1, "name": "Laptop", "quantity": 10},
    {"id": 2, "name": "Mouse", "quantity": 50},
    {"id": 3, "name": "Keyboard", "quantity": 30}
]


class InventoryDataDetails(BaseModel):
    id: int
    name: str
    quantity: int

class HelpfulAgentOutput(BaseModel):
    response_type: str | Any
    is_inventory_data: bool



@function_tool
def add_items(id: int, name: str, quantity: int):
    """Add a new item to the inventory"""
    inventory_data.append({"id": id, "name": name, "quantity": quantity})
    print("Add tool call")
    return f"Item '{name}' with quantity {quantity} added successfully!"


@function_tool
def remove_items(id: int, quantity: int):
    """Remove a certain quantity of an item"""
    for item in inventory_data:
        if item["id"] == id:
            if item["quantity"] >= quantity:
                item["quantity"] -= quantity
                return f"Removed {quantity} units from {item['name']}. Remaining: {item['quantity']}."
            else:
                return f"Not enough quantity to remove. Available: {item['quantity']}."
    print("remove item called--->")
    return f"Item with id {id} not found."


@function_tool
def update_items(id: int, name: str = None, quantity: int = None):
    """Update the name or quantity of an item"""
    for item in inventory_data:
        if item["id"] == id:
            if name:
                item["name"] = name
            if quantity is not None:
                item["quantity"] = quantity
            return f"Item with id {id} updated: {item}"
    print("Update item tool called ---->")
    return f"Item with id {id} not found."


@function_tool
def view_items() :
    """View all items in the inventory and check the inventory data details"""
    print("View tool fire--->")
    return inventory_data



agent = Agent(
    name="Inventory Manager",
    instructions="You are a helpful Agent capable of managing an inventory. Always use the tools (add_items, remove_items, update_items, view_items) for operations.",
    model=gemin_model,
    tools=[add_items, remove_items, update_items, view_items],
    # output_type=InventoryDataDetails
)

result = Runner.run_sync(
    agent,
    input="""
1. Add 3 computers at id 4.
2. Remove 2 laptops on id 1.
3. Update mouse quantity to 40 on id 2.
4. Show me the updated inventory.
"""
)

print(result.final_output)











