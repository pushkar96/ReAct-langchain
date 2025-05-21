from typing import Union

from dotenv import load_dotenv
from langchain.agents import tool
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import render_text_description, Tool
from langchain_google_vertexai import ChatVertexAI
from langchain.agents.format_scratchpad import format_log_to_str

from helper import getProjectName
from prompts.prompts import react_template
from callbacks import AgentCallbackHandler

load_dotenv()

@tool
def get_text_length(text: str) -> int:
    """
    Calculates and returns the length of the given text.

    This function determines the number of characters in the provided string
    and returns the count as an integer value. It excludes any structural or
    additional processing, focusing solely on computing the character
    count of the input text.

    :param text: The input string for which the length is to be calculated.
    :type text: str

    :return: The number of characters in the provided text.
    :rtype: int
    """
    text = text.strip("'\n'").strip('"')
    return len(text)


def find_tool_by_name(tools, tool_name) -> Tool:
    for t in tools:
        if t.name == tool_name:
            return t
    raise ValueError(f"Tool  with name {tool_name} not found")


if __name__ == "__main__":
    tools = [get_text_length]
    intermediate_steps = []

    prompt = PromptTemplate.from_template(template=react_template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join(t.name for t in tools),
    )

    llm = ChatVertexAI(
        temperature=0,
        model_name="gemini-2.0-flash-001",
        project=getProjectName(),
        stop=["\nObservation", "Observation"],
        callbacks=[AgentCallbackHandler()],
    )

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    agentStep = ""
    while not isinstance(agentStep, AgentFinish):
        agentStep: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of 'hello'?",
                "agent_scratchpad": intermediate_steps,
            }
        )
        print(agentStep)

        if isinstance(agentStep, AgentAction):
            tool_name = agentStep.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agentStep.tool_input

            observation = tool_to_use.func(str(tool_input))
            intermediate_steps.append((agentStep, str(observation)))
            print(observation)

        if isinstance(agentStep, AgentFinish):
            print("### AgentFinish ###")
            print(agentStep.return_values)

    # print(agentStep)

    # llm.run(prompt)
