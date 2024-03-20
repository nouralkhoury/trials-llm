import sys
import promptlayer
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
            llm = PromptLayerChatOpenAI(pl_tags=pl_tags,
                                        temperature=0,
                                        model=model_name)
        else:
            llm = ChatOpenAI(api_key=OPENAI_API_KEY,
                             model=model_name,
                             temperature=0)
        return llm

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

    def setup_gpt(self, prompt_template, model_name: str = "gpt-3.5-turbo", pl_tags: list = []):
        try:
            # llm setup
            llm = self.llm_setup(model_name, pl_tags)
        except Exception as e:
            print(f"Error setting up LLM {e}", file=sys.stderr)
            return
        try:
            # chain setup
            chain = self.chain_setup(llm, prompt_template)
        except Exception as e:
            print(f"Error setting up llm chain {e}", file=sys.stderr)
            return
        return chain
