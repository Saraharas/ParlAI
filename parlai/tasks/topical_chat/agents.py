#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
A dataset with conversations directly grounded with knowledge retrieved from Wikipedia.
Contains 201k utterances from 22k dialogues spanning over 1300 diverse topics, split
into train, test, and valid sets. The test and valid sets are split into two sets each:
one with overlapping topics with the train set, and one with unseen topics.

To access the different valid/test splits (unseen/seen), specify
the corresponding split (`random_split` for seen, `topic_split`
for unseen) after the last colon in the task.
E.g. `wizard_of_wikipedia:WizardDialogKnowledgeTeacher:random_split`
"""

import copy
from parlai.core.teachers import FixedDialogTeacher, MultiTaskTeacher
from parlai.utils.io import PathManager
#from .build import build

import json
import os
import random


TOKEN_NOCHOSEN = 'no_passages_used'
TOKEN_KNOWLEDGE = '__knowledge__'
TOKEN_END_KNOWLEDGE = '__endknowledge__'


def _first_val(dictionary):
    vals = list(dictionary.values())
    if len(vals) > 0:
        return vals[0]
    return ''


def _first_key(dictionary):
    keys = list(dictionary.keys())
    if len(keys) > 0:
        return keys[0]
    return ''


def _get_chosen_title_and_sent(wizard_entry, k_dict):
    """
    Return a nicely extracted title and chosen sentence.

    :return: pair (title, sentence)
    """
    title_dict = wizard_entry.get('checked_passage', 'none')
    sentence_dict = wizard_entry.get('checked_sentence', {})
    title = None
    sentence = None
    if sentence_dict == {}:
        title = sentence = TOKEN_NOCHOSEN
    else:
        sentence = _first_val(sentence_dict)
        if sentence == TOKEN_NOCHOSEN:
            title = TOKEN_NOCHOSEN
        else:
            title = ''
            # cand_title1 is the title from the `checked_passage`
            cand_title1 = _first_val(title_dict) if title_dict else ''
            # cand_title2 is the extracted title of the passage from the
            #   sentence dict, which is e.g. `self_Vermont_Syrup_0`
            cand_title2 = ' '.join(_first_key(sentence_dict).split('_')[-1])
            if (
                cand_title1
                and cand_title1 in k_dict
                and sentence in k_dict[cand_title1]
            ):
                title = cand_title1
            elif cand_title2 in k_dict and sentence in k_dict[cand_title2]:
                title = cand_title2
            else:  # neither candidate title is the right one
                for t, passage in k_dict.items():
                    if sentence in passage:
                        title = t
                        break

    return title, sentence


def _path(opt):
    suffix = 'train' 
    if opt['datatype'].startswith('valid'):
        suffix = 'valid_rare'
    if opt['datatype'].startswith('test'):
        suffix = 'test_rare'
    dp = os.path.join(opt['datapath'], opt['task'], suffix + '_parlai.json')
    print('loading ', dp)
    return dp


class TopicalChatTeacher(FixedDialogTeacher):
    """
    The default teacher; essentially reads the json file and outputs the raw data.

    Actions have the following form:
    {
        'wizard_eval': <evaluation of wizard>,
        'chosen_topic': <chosen_topic>,
        'chosen_topic_passage': <chosen topic passage>,
        'mtdo': <whether the conversation had sufficient overlap>,
        'text': <text>
        'retrieved_topics': <topics retrieved for text>
        'full_retrieved_passages': <full retrieved passages>
        'retrieved_passages': <passages shown to turker>
        'checked_sentence': <checked sentence if wizard, else None>
        'checked_passage': <checked_passage if wizard, else None>
    }

    The 'passages' are lists of 1 entry dicts, mapping a topic to the sentences

    Specify the valid/test split after the last colon in the task, e.g.
    wizard_of_wikipedia:<teacher>:random_split
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.opt = opt
        task = opt.get('task', 'topical_chat')
        #split = task.split(':')
        #split = split[2] if len(split) == 3 else 'random_split'
        if opt.get('data_regime', '') == 'interactive':
            self.data = {}
            print('interactive mode: no task data needed')
        else:
            self.opt['task'] = 'topical_chat'
            if shared and 'data' in shared:
                self.data = shared['data']
            else:
                self.data_path = _path(opt)
                self._setup_data()
        self.num_exs = sum(len(d['dialog']) for d in self.data)
        self.reset()

    def _setup_data(self):
        print('loading: ' + self.data_path)
        with PathManager.open(self.data_path) as f:
            self.data = json.load(f)
            print('Number of dialogues ', len(self.data), '\n')
            print('Number of utterances ', sum(len(d['dialog']) for d in self.data), '\n')


    def num_episodes(self):
        return len(self.data)

    def num_examples(self):
        return self.num_exs

    def get(self, episode_idx, entry_idx=0):
        d = self.data[episode_idx]
        dialog_entry = d['dialog'][entry_idx]
        episode_done = entry_idx == len(d['dialog']) - 1
        action = {
            #'wizard_eval': d['wizard_eval'],
            'chosen_topic': d['chosen_topic'],
            'chosen_topic_passage': d['chosen_topic_passage'],
            'text': dialog_entry['text'],
            'retrieved_topics': dialog_entry['retrieved_topics'],
            'retrieved_passages': dialog_entry['retrieved_passages'],
            'checked_sentence': dialog_entry.get('checked_sentence', None),
            'checked_passage': dialog_entry.get('checked_passage', None),
            'episode_done': episode_done,
        }

        return action

    def share(self):
        shared = super().share()
        shared['data'] = self.data
        return shared





class TopicalChatDialogKnowledgeTeacher(TopicalChatTeacher):
    """
    Teacher that returns the following action dict:
    {
        'text': chosen_topic\n # if first ex in ep
                last_apprentice_message\n # if possible
                wizard_message # if --label-type is chosen_sent

        'knowledge': title_1 sentence_1\n
                            .
                            .
                            .
                     title_m sentence_n # all knowledge available to wizard
        'labels': [title_checked sentence_checked] # default
                                    OR
                  [wizard_response] # if --label-type set to 'response'

        'label_candidates': knowledge + [no_passages_used no_passages_used]
    }
    """

    def __init__(self, opt, shared=None):
        super().__init__(opt, shared)
        self.label_type = opt.get('label_type', 'response')
        self.include_knowledge = opt.get('include_knowledge', True)
        self.include_checked_sentence = opt.get('include_checked_sentence', False)
        self.knowledge_separator = opt.get('include_knowledge_separator', False)
        self.chosen_topic_delimiter = opt.get('chosen_topic_delimiter', '\n')
        self.num_exs = sum(self.len_episode(i) for i in range(len(self.data)))

    @staticmethod
    def add_cmdline_args(argparser):
        agent = argparser.add_argument_group('TopicalChat Dialog Knowledge arguments')
        agent.add_argument(
            '--label-type',
            type=str,
            choices=['response', 'chosen_sent'],
            default='response',
            help='whether to populate label field with the '
            'wizard response, or the chosen sentence',
        )
        agent.add_argument(
            '--include-knowledge',
            type='bool',
            default=True,
            help='Whether to include the knowledge available to' ' the wizard',
        )
        agent.add_argument(
            '--include-checked-sentence',
            type='bool',
            default=True,
            help='Whether to include the Wizard\'s' 'checked sentence',
        )
        agent.add_argument(
            '--include-knowledge-separator',
            type='bool',
            default=False,
            help='include special __knowledge__ token between ' 'title and passage',
        )
        agent.add_argument(
            '--chosen-topic-delimiter',
            type=str,
            default='\n',
            help='delimiter used when including chosen topic',
        )
        agent.add_argument(
            '--num-topics',
            type=int,
            default=5,
            help='in interactive mode, this is the number of topic choices'
            'the human will have',
        )
    
    def getID(self):
        return "TopicalChat DefaultTeacher"
    
    def len_episode(self, ep):
        d = self.data[ep]
        #wizard_first = 'Wizard' in d['dialog'][0]['speaker']
        #if wizard_first:
        #    return (len(d['dialog']) - 1) // 2
        return len(d['dialog']) // 2

    def num_examples(self):
        return self.num_exs

    def get(self, episode_idx, entry_idx=0):
        d = self.data[episode_idx]
        episode_done = entry_idx == (self.len_episode(episode_idx) - 1)
        
        #always apprentice starts
        wizard_first = False #'Wizard' in d['dialog'][0]['speaker']
        idx = entry_idx * 2 if wizard_first else (entry_idx * 2) + 1

        # first, get knowledge
        apprentice_ret_passages = wizard_ret_passages = {}

        if not wizard_first or idx != 0:
            apprentice_entry = d['dialog'][idx - 1]
            apprentice_ret_passages = apprentice_entry['retrieved_passages']
        if idx - 2 >= 0:
            wizard_prev_entry = d['dialog'][idx - 2]
            wizard_ret_passages = wizard_prev_entry['retrieved_passages']

        chosen_topic = d.get('chosen_topic', '')
        chosen_topic_passages = d['chosen_topic_passage']
        chosen_topic = d.get('chosen_topic', '')
        if chosen_topic:
            knowledge_dict = {chosen_topic: chosen_topic_passages}
        else:
            # include current bot(aka wizard) knowledge that grounds the golden bot response
            wiz_entry = d['dialog'][idx]
            wiz_passages = wiz_entry['retrieved_passages']
            #[{topic: [fact_sentence]}]
            knowledge_dict = wiz_passages[0]
        
        for ret_passes in [apprentice_ret_passages, wizard_ret_passages]:
            for passage in ret_passes:
                for k, v in passage.items():
                    if k not in knowledge_dict.keys():
                        knowledge_dict[k] = v

        # then, get text
        if idx == 0:
            # first message - only have the chosen topic
            text = chosen_topic
        # no overall episode topic
        
        #elif idx == 1:
        #    # first response - only have the first message
        #    text = (
        #        f"{chosen_topic}{self.chosen_topic_delimiter}{apprentice_entry['text']}"
        #    )
        
        else:
            text = ''
            if self.label_type == 'chosen_sent':
                # if chosen_sent, add wizard response to dialog history
                text += '{}\n'.format(wizard_prev_entry['text'])
            text += apprentice_entry['text']

        # next, get label
        wizard_entry = d['dialog'][idx]
        if self.label_type == 'response':
            labels = [wizard_entry['text']]
        else:
            title, sentence = _get_chosen_title_and_sent(wizard_entry, knowledge_dict)
            if self.knowledge_separator and title != TOKEN_NOCHOSEN:
                labels = ['{} {} {}'.format(title, TOKEN_KNOWLEDGE, sentence)]
            else:
                labels = ['{} {}'.format(title, sentence)]

        # finally, get label_candidates
        label_cands = ['{} {}'.format(TOKEN_NOCHOSEN, TOKEN_NOCHOSEN)]
        knowledge_str = ''
        for title, passage in knowledge_dict.items():
            for p in passage:
                if self.knowledge_separator:
                    cand = '{} {} {}'.format(title, TOKEN_KNOWLEDGE, p)
                else:
                    cand = '{} {}'.format(title, p)
                knowledge_str += cand + '\n'
                label_cands.append(cand)
        if self.label_type == 'response':
            if 'train' in self.datatype:
                label_cands = []
            else:
                label_cands = wizard_entry.get('candidate_responses', [])

        action = {
            'id': 'TopicalChatDialogKnowledgeTeacher',
            'text': text,
            'labels': labels,
            'chosen_topic': chosen_topic,
            'episode_done': episode_done,
            'label_candidates': label_cands,
        }
        if self.include_knowledge:
            action['knowledge'] = knowledge_str
        if self.include_checked_sentence:
            title, sentence = _get_chosen_title_and_sent(wizard_entry, knowledge_dict)
            action['title'] = title
            action['checked_sentence'] = sentence
        return action



class GeneratorTeacher(TopicalChatDialogKnowledgeTeacher):
    """
    Teacher for training a generator.

    Depending on certain flag configurations, the teacher will include differing amounts
    of knowledge
    """

    def __init__(self, opt, shared=None):
        #print('Generator teacher is chosen')
        opt['label_type'] = 'response'
        opt['include_checked_sentence'] = True
        super().__init__(opt, shared)
        self.knowledge_separator = opt.get('include_knowledge_separator', True)
        self.only_checked_knowledge = opt.get('only_checked_knowledge', False)
        self.prepend_gold_knowledge = opt.get('prepend_gold_knowledge')
        self.gold_knowledge_delimiter = opt.get('gold_knowledge_delimiter', '\n')
        self.dropout = opt.get('ignorant_dropout', 0.0)

    @staticmethod
    def add_cmdline_args(argparser):
        argparser.set_defaults(include_knowledge_separator=True)
        TopicalChatDialogKnowledgeTeacher.add_cmdline_args(argparser)
        agent = argparser.add_argument_group('TopicalChatGeneratorTeacher Arguments')
        agent.add_argument(
            '--only-checked-knowledge',
            type='bool',
            default=False,
            help='If true, only the checked sentence is provided',
        )
        agent.add_argument(
            '--ignorant-dropout',
            type=float,
            default=0.0,
            help='Eliminate all knowledge with this probability.'
            'Specify 1 for completely ignorant teacher',
        )
        agent.add_argument(
            '--prepend-gold-knowledge',
            type='bool',
            default=False,
            help='If true, prepend text with checked sentence',
        )
        agent.add_argument(
            '--gold-knowledge-delimiter',
            type=str,
            default='\n',
            help='delimiter for prepending gold knowledge',
        )

    def getID(self):
        return "TopicalChat GeneratorTeacher"

    def get(self, episode_idx, entry_idx=0):
        a = super().get(episode_idx, entry_idx)
        # zero out the label candidates?
        if 'knowledge' not in a:
            # just a batch padding item
            return a
        # save some memory, we don't need label_candidates
        a['label_candidates'] = []
        
        #if not a['knowledge'].startswith(TOKEN_NOCHOSEN):
            # make sure the token is appearing
        #    a['knowledge'] = (
        #        TOKEN_NOCHOSEN
        #        + ' '
        #        + TOKEN_KNOWLEDGE
        #        + ' '
        #        + TOKEN_NOCHOSEN
        #        + '\n'
        #        + a['knowledge']
        #    )
        if self.only_checked_knowledge:
            # useful for test time evaluation, where it's only ever trained on true
            # knowledge
            a['knowledge'] = (
                a['title'] + ' ' + TOKEN_KNOWLEDGE + ' ' + a['checked_sentence']
            )

        if random.random() < self.dropout:
            # Drop the knowledge with some probability
            a['title'] = TOKEN_NOCHOSEN
            a['checked_sentence'] = TOKEN_NOCHOSEN
            a['knowledge'] = (
                TOKEN_NOCHOSEN + ' ' + TOKEN_KNOWLEDGE + ' ' + TOKEN_NOCHOSEN
            )
        elif self.prepend_gold_knowledge:
            a[
                'text'
            ] = f"{TOKEN_KNOWLEDGE} {a['checked_sentence']} {TOKEN_END_KNOWLEDGE}{self.gold_knowledge_delimiter}{a['text']}"
        #print('sample ', a['text'])
        return a
class DefaultTeacher(TopicalChatDialogKnowledgeTeacher):
    pass