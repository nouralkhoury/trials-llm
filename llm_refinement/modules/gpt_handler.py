import sys
import promptlayer
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chat_models import PromptLayerChatOpenAI
from configurations.config import OPENAI_API_KEY, PROMPTLAYER_API_KEY


class GPTHandler(object):

    @staticmethod
    def llm_setup(model_name: str = "gpt-3.5-turbo", pl_tags: list = []):
        """
        Sets up a language model.

        Args:
        model_name: The name of the model to use.
        pl_tags: PromptLayer tags. If [], promptLayer will not be used.

        Returns:
        ChatOpenAI: A language model object.
        """
        if pl_tags:
            promptlayer.api_key = PROMPTLAYER_API_KEY
            llm = PromptLayerChatOpenAI(pl_tags=["zero-shot-gpt-4"],
                                        openai_api_key=OPENAI_API_KEY,
                                        temperature=0,
                                        model=model_name)
        else:
            llm = ChatOpenAI(api_key=OPENAI_API_KEY,
                             model=model_name,
                             temperature=0)
        return llm

    @staticmethod
    def prompt_setup(template: str, num_examples: int = 0) -> dict:
        """
        Sets up a prompt template.

        Args:
        template: The template prompt to be used
        num_examples: The number of examples to be used in few-shot. Default: 0 (zero-shot)

        Returns:
        dict: A dictionary containing the prompt template.
        """
        try:
            if not isinstance(num_examples, int):    
                raise TypeError("Invalid 'num_examples'. Expected an integer.")

            if not isinstance(template, str):
                raise TypeError("Invalid 'template'. Expected a string.")
        except TypeError as e:
            print(f"Prompt setup failed: {e}")

        if num_examples == 0:
            prompt_template = PromptTemplate(
                input_variables=["trial"],
                template=template)
        else:
            # setup the example variable
            examples = [f"example_{i}" if num_examples > 1 else "example" for i in range(1, num_examples + 1)]
            prompt_template = PromptTemplate(
                input_variables=["trial"] + examples,
                template=template)
        return prompt_template

    @staticmethod
    def chain_setup(llm, prompt_template):
        """
        Sets up a LLM chain

        Args:
        llm: The language model to use.

        Returns:
        LLMChain: An LLMChain object
        """
        chain = LLMChain(llm=llm, prompt=prompt_template)
        return chain

    def setup_gpt(self, model_name: str = "gpt-3.5-turbo", template=str, num_examples: int = 0, pl_tags: list = []):
        try:
            # llm setup
            llm = self.llm_setup(model_name, pl_tags)
        except Exception as e:
            print(f"Error setting up LLM {e}", file=sys.stderr)
            return

        try:
            # prompt setup
            prompt_template = self.prompt_setup(template, num_examples)
        except Exception as e:
            print(f"Error setting up prompt template {e}", file=sys.stderr)
            return

        try:
            # chain setup
            chain = self.chain_setup(llm, prompt_template)
        except Exception as e:
            print(f"Error setting up llm chain {e}", file=sys.stderr)
            return
        return chain
