# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

import os
# DeepSpeed Team
from datasets import load_dataset, load_from_disk
from torch.utils.data import Subset
import re


# The template prompt dataset class that all new dataset porting needs to
# follow in order to have a unified API and unified data format.
class PromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        if os.path.exists(dataset_name):
            self.raw_datasets = load_from_disk(dataset_name)
        elif not dataset_name == 'local/jsonfile':
            self.raw_datasets = load_dataset(dataset_name)


    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return

class LoadPromptRawDataset(object):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name=None):
        self.output_path = output_path
        self.seed = seed
        self.local_rank = local_rank
        if test_dataset_name == None:
            self.raw_datasets = load_dataset('parquet', data_dir=dataset_name, data_files={'train': train_dataset_name})
        else:
            self.raw_datasets = load_dataset('parquet',data_dir=dataset_name,data_files={'train':train_dataset_name,'test':test_dataset_name})


    def get_train_data(self):
        return

    def get_eval_data(self):
        return

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        return

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        return

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        return

    def get_prompt_and_chosen(self, sample):
        return

    def get_prompt_and_rejected(self, sample):
        return


# English dataset
class DahoasRmstaticDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Dahoas/rm-static"
        self.dataset_name_clean = "Dahoas_rm_static"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class LoadDahoasRmstaticDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "Dahoas/rm-static"
        self.dataset_name_clean = "Dahoas_rm_static"

    def get_train_data(self):
        return self.raw_datasets["train"]

    # eval or test???????
    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class LoadDahoasRmstaticRelabeledLlama2Dataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "Dahoas/rm-static-relabeled_Llama2"
        self.dataset_name_clean = "Dahoas_rm_static_relabeled_Llama2"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


# English dataset
class DahoasFullhhrlhfDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Dahoas/full-hh-rlhf"
        self.dataset_name_clean = "Dahoas_full_hh_rlhf"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class LoadDahoasFullhhrlhfDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "Dahoas/full-hh-rlhf"
        self.dataset_name_clean = "Dahoas_full_hh_rlhf"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']
 
# Translated datasets
class LoadDahoasFullhhrlhfzhDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/full-hh-rlhf-zh"
        self.dataset_name_clean = "PCL_full_hh_rlhf_zh"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['zh_prompt']

    def get_chosen(self, sample):
        return sample['zh_chosen']

    def get_rejected(self, sample):
        return sample['zh_rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['zh_prompt'] + sample['zh_chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['zh_prompt'] + sample['zh_rejected']
    
class LoadDahoasSinglehhrlhfzhDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/single-hh-rlhf-zh"
        self.dataset_name_clean = "PCL_single_hh_rlhf_zh"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['zh_prompt']

    def get_chosen(self, sample):
        return sample['zh_chosen']

    def get_rejected(self, sample):
        return sample['zh_rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['zh_prompt'] + sample['zh_chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['zh_prompt'] + sample['zh_rejected']
    
class LoadPCLComplianceZhDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/compliance_zh"
        self.dataset_name_clean = "PCL_compliance_zh"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class LoadPCLComplianceZhCleanDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/compliance_zh_clean"
        self.dataset_name_clean = "PCL_compliance_zh_clean"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']
    
class LoadPCLRedteamingZhDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/red_teaming_zh"
        self.dataset_name_clean = "PCL_red_teaming_zh"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']
    
    def get_score(self, sample):
        return sample['margin']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class LoadPCLRedteamingZhFourDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/red_teaming_4zh"
        self.dataset_name_clean = "PCL_red_teaming_4zh"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']
    
    def get_score(self, sample):
        return sample['margin']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']
    
class LoadPCLRedteamingZhFiveDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/red_teaming_5zh"
        self.dataset_name_clean = "PCL_red_teaming_5zh"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']
    
    def get_score(self, sample):
        return sample['margin']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class LoadPCLRedteamingZhFiveCleanDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/red_teaming_5_clean_zh"
        self.dataset_name_clean = "PCL_red_teaming_5_cleanzh"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']
    
    def get_score(self, sample):
        return sample['margin']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class LoadPCLRedteamingZhSixDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/red_teaming_0908zh"
        self.dataset_name_clean = "PCL_red_teaming_0908zh"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']
    
    def get_score(self, sample):
        return sample['margin']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']
    
class LoadPCLRedteamingZhSevenDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/red_teaming_0915zh"
        self.dataset_name_clean = "PCL_red_teaming_0915zh"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']
    
    def get_score(self, sample):
        return sample['margin']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']
    
class LoadPCLRedteamingPairs1019ZhDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/red_teaming_pairs_1019zh"
        self.dataset_name_clean = "PCL_red_teaming_pairs_1019zh"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']
    
    def get_score(self, sample):
        return sample['margin']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class LoadPCLRedteamingZhSevenCleanDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/red_teaming_0915Cleanzh"
        self.dataset_name_clean = "PCL_red_teaming_0915Cleanzh"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']
    
    def get_score(self, sample):
        return sample['margin']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']
    
class LoadPCLRedteamingZhSevenSixDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/red_teaming_0915N08zh"
        self.dataset_name_clean = "PCL_red_teaming_0915N08zh"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']
    
    def get_score(self, sample):
        return sample['margin']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']
    
class LoadPCLRedteamingZhFiveCleanDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/red_teaming_5zh_clean"
        self.dataset_name_clean = "PCL_red_teaming_5zh_clean"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']
    
    def get_score(self, sample):
        return sample['margin']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class LoadInstructionDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/instruction"
        self.dataset_name_clean = "PCL_instruction"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

class LoadInstruction3wDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/instruction_3w"
        self.dataset_name_clean = "PCL_instruction_3w"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']


class LoadComplianceRSDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/ComplianceRS"
        self.dataset_name_clean = "PCL_ComplianceRS"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

class LoadComplianceRS0901Dataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/ComplianceRS0901"
        self.dataset_name_clean = "PCL_ComplianceRS0901"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']
    
class LoadRedTeamingRSV0Dataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/RedTeamingRS_V0"
        self.dataset_name_clean = "PCL_RedTeamingRS_V0"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

class LoadRedTeamingRS0901Dataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/RedTeamingRS_0901"
        self.dataset_name_clean = "PCL_RedTeamingRS_0901"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']
    
class LoadRedTeamingRS0915Dataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/RedTeamingRS_0915"
        self.dataset_name_clean = "PCL_RedTeamingRS_0915"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

class Loadphase1_originalRedTeamingDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/phase1_originalRedTeaming"
        self.dataset_name_clean = "PCL_phase1_originalRedTeaming"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class Loadphase1_phase2_phase3Dataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/phase1_phase2_phase3"
        self.dataset_name_clean = "PCL_phase1_phase2_phase3"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class Loadphase1_phase3Dataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "PCL/phase1_phase3"
        self.dataset_name_clean = "PCL_phase1_phase3"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']



class LoadOpenAISummarizationDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "OpenAI/Summarization"
        self.dataset_name_clean = "OpenAI_Summarization"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + "\n" + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + "\n" + sample['rejected']
    
class LoadOpenAISummarizationRelabelDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "OpenAI/Summarization_Relabel"
        self.dataset_name_clean = "OpenAI_Summarization_Relabel"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + "\n" + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + "\n" + sample['rejected']

class LoadOpenAISummarizationRelabel25NoiseDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "OpenAI/Summarization_25Relabel_noise"
        self.dataset_name_clean = "OpenAI_Summarization_25Relabel_noise"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + "\n" + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + "\n" + sample['rejected']
    
class LoadOpenAISummarizationRelabel15NoiseDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "OpenAI/Summarization_15Relabel_noise"
        self.dataset_name_clean = "OpenAI_Summarization_15Relabel_noise"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['chosen']

    def get_rejected(self, sample):
        return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + "\n" + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + "\n" + sample['rejected']




# English dataset
class DahoasSyntheticinstructgptjpairwiseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Dahoas/synthetic-instruct-gptj-pairwise"
        self.dataset_name_clean = "Dahoas_synthetic_instruct_gptj_pairwise"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        return " Human: " + sample['prompt'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['chosen']

    def get_rejected(self, sample):
        return " " + sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['prompt'] + " Assistant: " + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return " Human: " + sample['prompt'] + " Assistant: " + sample[
            'rejected']

class LoadDahoasSyntheticinstructgptjpairwiseDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "Dahoas/synthetic-instruct-gptj-pairwise"
        self.dataset_name_clean = "Dahoas_synthetic_instruct_gptj_pairwise"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        return " Human: " + sample['prompt'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['chosen']

    def get_rejected(self, sample):
        return " " + sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['prompt'] + " Assistant: " + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return " Human: " + sample['prompt'] + " Assistant: " + sample[
            'rejected']


class LoadDahoasSftstaticDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name=None):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "Dahoas/sft-static"
        self.dataset_name_clean = "Dahoas_sft_static"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    # def get_train_data(self):
    #     return self.raw_datasets["train"]
    #
    # def get_eval_data(self):
    #     if "test" in self.raw_datasets.keys():
    #         return self.raw_datasets["test"]
    #     else:
    #         return None

    def get_prompt(self, sample):
        return sample['prompt']

    def get_chosen(self, sample):
        return sample['response']

    # def get_rejected(self, sample):
    #     return sample['rejected']

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['response']

    # def get_prompt_and_rejected(self, sample):
    #     return sample['prompt'] + sample['rejected']



# English dataset
class YitingxieRlhfrewarddatasetsDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "yitingxie/rlhf-reward-datasets"
        self.dataset_name_clean = "yitingxie_rlhf_reward_datasets"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt'] + "Assistant:"

    def get_chosen(self, sample):
        return sample['chosen'].split("Assistant:")[-1]

    def get_rejected(self, sample):
        return sample['rejected'].split("Assistant:")[-1]

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']

class LoadYitingxieRlhfrewarddatasetsDataset(LoadPromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name, train_dataset_name, test_dataset_name)
        self.dataset_name = "yitingxie/rlhf-reward-datasets"
        self.dataset_name_clean = "yitingxie_rlhf_reward_datasets"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        return sample['prompt'] + "Assistant:"

    def get_chosen(self, sample):
        return sample['chosen'].split("Assistant:")[-1]

    def get_rejected(self, sample):
        return sample['rejected'].split("Assistant:")[-1]

    def get_prompt_and_chosen(self, sample):
        return sample['prompt'] + sample['chosen']

    def get_prompt_and_rejected(self, sample):
        return sample['prompt'] + sample['rejected']


# English dataset
class OpenaiWebgptcomparisonsDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "openai/webgpt_comparisons"
        self.dataset_name_clean = "openai_webgpt_comparisons"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        return " Human: " + sample['question']['full_text'] + " Assistant:"

    def get_chosen(self, sample):
        if float(sample['score_0']) >= float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        # This data has citation square brackets and numbers (e.g., "[1]").
        # Right now we are not doing browser-assisted finetuning, thus we
        # remove these citations to avoid confusing the model.
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " " + response

    def get_rejected(self, sample):
        if float(sample['score_0']) < float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " " + response

    def get_prompt_and_chosen(self, sample):
        if float(sample['score_0']) >= float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " Human: " + sample['question'][
            'full_text'] + " Assistant: " + response

    def get_prompt_and_rejected(self, sample):
        if float(sample['score_0']) < float(sample['score_1']):
            response = sample['answer_0']
        else:
            response = sample['answer_1']
        response = re.sub(r" [\(\[].*?[\)\]]", "", response)
        response = re.sub(r"[\(\[].*?[\)\]]", "", response)
        return " Human: " + sample['question'][
            'full_text'] + " Assistant: " + response


# English dataset
class StanfordnlpSHPDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "stanfordnlp/SHP"
        self.dataset_name_clean = "stanfordnlp_SHP"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample['history'] + " Assistant:"

    def get_chosen(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_A"]
        else:
            response = sample["human_ref_B"]
        return " " + response

    def get_rejected(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_B"]
        else:
            response = sample["human_ref_A"]
        return " " + response

    def get_prompt_and_chosen(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_A"]
        else:
            response = sample["human_ref_B"]
        return " Human: " + sample['history'] + " Assistant: " + response

    def get_prompt_and_rejected(self, sample):
        if int(sample["labels"]) == 1:
            response = sample["human_ref_B"]
        else:
            response = sample["human_ref_A"]
        return " Human: " + sample['history'] + " Assistant: " + response


# English dataset
class PvduySharegptalpacaoavicunaformatDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "pvduy/sharegpt_alpaca_oa_vicuna_format"
        self.dataset_name_clean = "pvduy_sharegpt_alpaca_oa_vicuna_format"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["test"]

    def get_prompt(self, sample):
        if sample['prompt'] is not None and len(sample['prompt']) > 0:
            return sample['prompt'].replace("USER", "Human").replace(
                "ASSISTANT", "Assistant")
        return None

    def get_chosen(self, sample):
        if sample['label'] is not None and len(sample['label']) > 0:
            return " " + sample['label']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['prompt'] is not None and sample['label'] is not None and len(
                sample['prompt']) > 0 and len(sample['label']) > 0:
            return sample['prompt'].replace("USER", "Human").replace(
                "ASSISTANT", "Assistant") + " " + sample['label']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


class LocalJsonFileDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name, chat_path):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "local/jsonfile"
        self.dataset_name_clean = "jsonfile"
        self.raw_datasets = load_dataset('json',
                                         data_files={
                                             "train":
                                             chat_path + '/data/train.json',
                                             "eval":
                                             chat_path + '/data/eval.json'
                                         })

    def get_train_data(self):
        if self.raw_datasets['train'] is not None:
            return self.raw_datasets['train']
        return None

    def get_eval_data(self):
        if self.raw_datasets['eval'] is not None:
            return self.raw_datasets['eval']
        return None

    # The prompt should be in the format of: " Human: " + actual_prompt_sentence + " Assistant:"
    def get_prompt(self, sample):
        if sample['prompt'] is not None:
            return " " + sample['prompt']
        return None

    # The chosen response should be in the format of: " " + actual_response_sentence
    def get_chosen(self, sample):
        if sample['chosen'] is not None:
            return " " + sample['chosen']
        return None

    # The rejected response should be in the format of: " " + actual_response_sentence
    # If the dataset does not have rejected response, return None
    def get_rejected(self, sample):
        if sample['rejected'] is not None:
            return " " + sample['rejected']
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['prompt'] is not None and sample['chosen'] is not None:
            return " " + sample['prompt'] + " " + sample['chosen']
        return None

    def get_prompt_and_rejected(self, sample):
        if sample['prompt'] is not None and sample['rejected'] is not None:
            return " " + sample['prompt'] + " " + sample['rejected']
        return None


# Chinese dataset
class Wangrui6ZhihuKOLDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "wangrui6/Zhihu-KOL"
        self.dataset_name_clean = "wangrui6_Zhihu_KOL"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['INSTRUCTION'] is not None:
            return " Human: " + sample['INSTRUCTION'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['RESPONSE'] is not None:
            return " " + sample['RESPONSE']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['INSTRUCTION'] is not None and sample['RESPONSE'] is not None:
            return " Human: " + sample[
                'INSTRUCTION'] + " Assistant: " + sample['RESPONSE']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Chinese dataset
class CohereMiraclzhqueries2212Dataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Cohere/miracl-zh-queries-22-12"
        self.dataset_name_clean = "Cohere_miracl_zh_queries_22_12"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["dev"]

    def get_prompt(self, sample):
        return " Human: " + sample['query'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['positive_passages'][0]['text']

    def get_rejected(self, sample):
        return " " + sample['negative_passages'][0]['text']

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['query'] + " Assistant: " + sample[
            'positive_passages'][0]['text']

    def get_prompt_and_rejected(self, sample):
        return " Human: " + sample['query'] + " Assistant: " + sample[
            'negative_passages'][0]['text']


# Chinese dataset
class HelloSimpleAIHC3ChineseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Hello-SimpleAI/HC3-Chinese"
        self.dataset_name_clean = "Hello_SimpleAI_HC3_Chinese"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['question'] is not None:
            return " Human: " + sample['question'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['human_answers'][0] is not None:
            return " " + sample['human_answers'][0]
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['question'] is not None and sample['human_answers'][
                0] is not None:
            return " Human: " + sample['question'] + " Assistant: " + sample[
                'human_answers'][0]
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Chinese dataset
class MkqaChineseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "mkqa-Chinese"
        self.dataset_name_clean = "mkqa"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['queries']['zh_cn'] is not None:
            return " Human: " + sample['queries']['zh_cn'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['answers']['zh_cn'][0]['text'] is not None:
            return " " + sample['answers']['zh_cn'][0]['text']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['queries']['zh_cn'] is not None and sample['answers'][
                'zh_cn'][0]['text'] is not None:
            return " Human: " + sample['queries'][
                'zh_cn'] + " Assistant: " + sample['answers']['zh_cn'][0][
                    'text']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Japanese dataset
class MkqaJapaneseDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "mkqa-Japanese"
        self.dataset_name_clean = "mkqa"

    def get_train_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 0,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_eval_data(self):
        from .data_utils import get_raw_dataset_split_index
        dataset = self.raw_datasets["train"]
        index = get_raw_dataset_split_index(self.local_rank, self.output_path,
                                            self.dataset_name_clean,
                                            self.seed, "train_eval", "9,1", 1,
                                            len(dataset))
        dataset = Subset(dataset, index)
        return dataset

    def get_prompt(self, sample):
        if sample['queries']['ja'] is not None:
            return " Human: " + sample['queries']['ja'] + " Assistant:"
        return None

    def get_chosen(self, sample):
        if sample['answers']['ja'][0]['text'] is not None:
            return " " + sample['answers']['ja'][0]['text']
        return None

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        if sample['queries']['ja'] is not None and sample['answers']['ja'][0][
                'text'] is not None:
            return " Human: " + sample['queries'][
                'ja'] + " Assistant: " + sample['answers']['ja'][0]['text']
        return None

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Japanese dataset
class CohereMiracljaqueries2212Dataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "Cohere/miracl-ja-queries-22-12"
        self.dataset_name_clean = "Cohere_miracl_ja_queries_22_12"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["dev"]

    def get_prompt(self, sample):
        return " Human: " + sample['query'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['positive_passages'][0]['text']

    def get_rejected(self, sample):
        return " " + sample['negative_passages'][0]['text']

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['query'] + " Assistant: " + sample[
            'positive_passages'][0]['text']

    def get_prompt_and_rejected(self, sample):
        if len(sample['negative_passages']) > 0:
            return " Human: " + sample['query'] + " Assistant: " + sample[
                'negative_passages'][0]['text']
        return None


# Japanese dataset
class LmqgQgjaquadDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "lmqg/qg_jaquad"
        self.dataset_name_clean = "lmqg_qg_jaquad"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample['question'] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['sentence']

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['question'] + " Assistant: " + sample[
            'sentence']

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None


# Japanese dataset
class LmqgQagjaquadDataset(PromptRawDataset):

    def __init__(self, output_path, seed, local_rank, dataset_name):
        super().__init__(output_path, seed, local_rank, dataset_name)
        self.dataset_name = "lmqg/qag_jaquad"
        self.dataset_name_clean = "lmqg_qag_jaquad"

    def get_train_data(self):
        return self.raw_datasets["train"]

    def get_eval_data(self):
        return self.raw_datasets["validation"]

    def get_prompt(self, sample):
        return " Human: " + sample['questions'][0] + " Assistant:"

    def get_chosen(self, sample):
        return " " + sample['paragraph']

    def get_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None

    def get_prompt_and_chosen(self, sample):
        return " Human: " + sample['questions'][0] + " Assistant: " + sample[
            'paragraph']

    def get_prompt_and_rejected(self, sample):
        print(
            f"Warning: dataset {self.dataset_name} does not include rejected response."
        )
        return None
