import asyncio
from dataclasses import dataclass
import os
from pydantic import BaseModel
from dotenv import load_dotenv
from agents import Agent, Runner, set_tracing_disabled, OpenAIChatCompletionsModel, function_tool, AsyncOpenAI, enable_verbose_stdout_logging

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
set_tracing_disabled(disabled=True)
enable_verbose_stdout_logging()

provider = AsyncOpenAI(
    api_key=api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)


gemin_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=provider
)


inventory = [
    {"id": 1, "name": "Laptop", "quantity": 10},
    {"id": 2, "name": "Mouse", "quantity": 50},
    {"id": 3, "name": "Keyboard", "quantity": 30}
]


@dataclass
class InventoryItemInput:
    operation: str
    id: int = None
    name: str = None
    quantity: int = None


class HelpfulAgentOutput(BaseModel):
    response_type: str  
    inventory_data: str = None


@function_tool
async def manageInventory(item: InventoryItemInput) -> str: 
    global inventory
    operation = item.operation.lower()

    if operation == "add":
        if not item.name or item.quantity is None:
            return "Error: Name and quantity are required for adding an item."
        new_id = max([inv["id"] for inv in inventory], default=0) + 1
        inventory.append({"id": new_id, "name": item.name, "quantity": item.quantity})
        return f"Added {item.name} with ID {new_id} and quantity {item.quantity}."

    elif operation == "update":
        if item.id is None or not item.name or item.quantity is None:
            return "Error: ID, name, and quantity are required for updating an item."
        for inv_item in inventory:
            if inv_item["id"] == item.id:
                inv_item["name"] = item.name
                inv_item["quantity"] = item.quantity
                return f"Updated item ID {item.id} to {item.name} with quantity {item.quantity}."
        return f"Error: Item with ID {item.id} not found."

    elif operation == "delete":
        if item.id is None:
            return "Error: ID is required for deleting an item."
        for i, inv_item in enumerate(inventory):
            if inv_item["id"] == item.id:
                deleted_item = inventory.pop(i)
                return f"Deleted item ID {item.id} ({deleted_item['name']})."
        return f"Error: Item with ID {item.id} not found."

    elif operation == "view":
        if item.id: 
            for inv_item in inventory:
                if inv_item["id"] == item.id:
                    return f"Item ID {inv_item['id']}: {inv_item['name']} (Qty: {inv_item['quantity']})"
            return f"Error: Item with ID {item.id} not found."
        else:  
            if not inventory:
                return "Inventory is empty."
            return "Current Inventory:\n" + "\n".join(
                [f"ID {inv['id']}: {inv['name']} (Qty: {inv['quantity']})" for inv in inventory]
            )

    else:
        return "Error: Invalid operation. Use 'add', 'update', 'delete', or 'view'."


agent = Agent(
    name="Helpful Assistant",
    instructions="You are a helpful Assistant capable of managing an inventory. Always use manageInventory for add, update, delete, and view operations.",
    model=gemin_model,
    tools=[manageInventory],
    output_type=HelpfulAgentOutput 
)

# Runner
async def main(kickOffMessage: str):
    print(f"RUN Initiated: {kickOffMessage}")
    
    result = await Runner.run(
        agent,
        input=kickOffMessage
    )
    print("\nAgent Output:", result.final_output)

   
    if result.final_output and result.final_output.response_type == "inventory":
        print("\nCurrent Inventory:")
        for item in inventory:
            print(item)

def start():
   
    asyncio.run(main("View all items in the inventory"))
    # Example: View single item
    asyncio.run(main("""
1. Add a new item to the inventory: Monitor with quantity 20.  
2. Delete 5 units of Laptop from the inventory (not the entire item).  
3. Show the updated inventory with all items and their remaining quantities.

"""))

if __name__ == "__main__":
    start()
