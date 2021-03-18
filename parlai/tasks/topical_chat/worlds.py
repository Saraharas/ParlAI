#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

#COURIER AGENT RELATED WORLD

from copy import deepcopy
import json
import os
import string

from parlai.core.agents import create_agent
from parlai.core.message import Message
from parlai.core.worlds import World, validate
from parlai.tasks.wizard_of_wikipedia.agents import TOKEN_KNOWLEDGE, TOKEN_END_KNOWLEDGE


class GeneratorWorld(World):
    
    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        print('Gen world activated')
        self.opt = opt
        self.teacher_agent = self.agents[0]
        self.model_agent = self.agents[1]   

    def parley_one_sample(self, user_input):
        print('one sample parley')
        #user_input = {'topic': str, 'knowledge': str, 'text': str, 'history': [str, str,..]}
        for utterance in user_input['history']:
            self.model_agent.history.add_reply(utterance.strip())

        try:
            act = Message()
            act['text'] = f"{TOKEN_KNOWLEDGE} {user_input['checked_sentence']} {TOKEN_END_KNOWLEDGE}{self.opt['gold_knowledge_delimiter']}{user_input['text']}"
            act['episode_done'] = True
        except StopIteration:
            print('[ EPISODE CHAT DONE ]')    
            self.model_agent.reset()
            return
        # model observes knowledge and human (apprentice) act
        self.model_agent.observe(validate(act))
        return self.model_agent.act()


    def parley(self, user_input, knowledge_key):
        # with batching
        print(f"batching parley with {knowledge_key} facts")
        #user_input = {'inputs': [{'topic': str, 'knowledge': str, 'text': str}, {'topic': str, 'knowledge': str, 'text': str} ...], 'history': [str, str,..]}
        try:
            batch_act = []
            for inp in user_input['inputs']:
                self.model_agent.history.reset()
                for utterance in user_input['history']:
                    self.model_agent.history.add_reply(utterance.strip())
        
                act = Message()
                act['text'] = f"{TOKEN_KNOWLEDGE} {inp[knowledge_key]} {TOKEN_END_KNOWLEDGE}{self.opt['gold_knowledge_delimiter']}{inp['text']}"
                act['episode_done'] = False
                
                batch_act.append(self.model_agent.observe(validate(act)))

        except StopIteration:
            print('[ EPISODE CHAT DONE ]')    
            self.model_agent.reset()
            return
        # model observes knowledge and human (apprentice) act
        #print(batch_act)
        return self.model_agent.batch_act(batch_act)


    
class InteractiveGeneratorWorld(World):
    
    def __init__(self, opt, agents, shared=None):
        super().__init__(opt, agents, shared)
        print('interactive world activated')
        self.opt = opt
        self.model_agent = self.agents[0]
        
    def parley(self, user_input):
        #user_input = {'topic': str, 'knowledge': str, 'text': str, 'history': [str, str,..]}
        for utterance in user_input['history']:
            self.model_agent.history.add_reply(utterance.strip())
        try:
            act = Message()
            act['id'] = 'TopicalChat GeneratorTeacher'
            act['text'] = f"{TOKEN_KNOWLEDGE} {user_input['knowledge'].strip()} {TOKEN_END_KNOWLEDGE}{self.opt['gold_knowledge_delimiter']}{user_input['text']}"
            act['episode_done'] = True
        except StopIteration:
            print('[ EPISODE CHAT DONE ]')    
            self.model_agent.reset()
            return
        # model observes knowledge and human (apprentice) act
        self.model_agent.observe(validate(act))
        return self.model_agent.act()
