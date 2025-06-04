import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset, DataLoader
from utils.parse import args
from typing import List
import os

class TextDataset(Dataset):
    def __init__(self, input_text: List[str]):
        self.input_text = input_text

    def __len__(self):
        return len(self.input_text)

    def __getitem__(self, idx):
        return self.input_text[idx]


class DataHandler:
    def __init__(self):
        if args.dataset == "amazon" or args.dataset == 're_amazon':
            self.system_prompt = "Explain why the user would buy with the book within 50 words."
            self.item = "book"
        elif args.dataset == "yelp" or args.dataset == "google" or args.dataset == 're_google' or args.dataset == 're_yelp':
            self.system_prompt = "Explain why the user would enjoy the business within 50 words."
            self.item = "business"
        
        if not args.random_random:
            if not args.random_embeddings:
                user_path = f"./data/{args.dataset}/user_emb_{args.graph}.pkl"
                item_path = f"./data/{args.dataset}/item_emb_{args.graph}.pkl"
                with open(user_path, "rb") as file:
                    self.user_emb = pickle.load(file)
                with open(item_path, "rb") as file:
                    self.item_emb = pickle.load(file)
            else:
                user_path = f'./data/{args.dataset}/user_emb_{args.out_name}.pkl'
                item_path = f'./data/{args.dataset}/item_emb_{args.out_name}.pkl'
                if os.path.isfile(user_path):
                    self.user_emb = torch.load(user_path)
                    self.item_emb = torch.load(item_path)
                else:
                    with open(f'./data/{args.dataset}/para_dict.pickle', 'rb') as f:
                        data_info = pickle.load(f)
                    self.user_emb = nn.init.xavier_uniform_(torch.empty(data_info['user_num'], 64))
                    self.item_emb = nn.init.xavier_uniform_(torch.empty(data_info['item_num'], 64))
                    torch.save(self.user_emb, f'./data/{args.dataset}/user_emb_{args.out_name}.pkl')
                    torch.save(self.item_emb, f'./data/{args.dataset}/item_emb_{args.out_name}.pkl')

    def load_data(self):
        # load data from data_loaders in data
        with open(f"./data/{args.dataset}/trn.pkl", "rb") as file:
            trn_data = pickle.load(file)
        with open(f"./data/{args.dataset}/val.pkl", "rb") as file:
            val_data = pickle.load(file)
        with open(f"./data/{args.dataset}/tst.pkl", "rb") as file:
            tst_data = pickle.load(file)

        # convert data into dictionary
        trn_dict = trn_data.to_dict("list")
        val_dict = val_data.to_dict("list")
        tst_dict = tst_data.to_dict("list")

        # combine all information input input string
        trn_input = []
        val_input = []
        tst_input = []
        
        if args.random_random:
            for i in range(len(trn_dict["uid"])):
                user_message = f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> {self.item} name: {trn_dict['title'][i]} <EXPLAIN_POS> {trn_dict['explanation'][i]}"
                trn_input.append(
                    (
                        f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]"
                    )
                )
            for i in range(len(val_dict["uid"])):
                user_message = f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> {self.item} name: {val_dict['title'][i]} <EXPLAIN_POS>"
                val_input.append(
                    (
                        f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]",
                        val_dict['explanation'][i],
                    )
                )
            for i in range(len(tst_dict["uid"])):
                user_message = f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> {self.item} name: {tst_dict['title'][i]} <EXPLAIN_POS>"
                tst_input.append(
                    (
                        f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]",
                        tst_dict["explanation"][i],
                    )
                )
        elif args.no_profile:
            for i in range(len(trn_dict["uid"])):
                user_message = f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> {self.item} name: {trn_dict['title'][i]} <EXPLAIN_POS> {trn_dict['explanation'][i]}"
                trn_input.append(
                    (
                        self.user_emb[trn_dict["uid"][i]],
                        self.item_emb[trn_dict["iid"][i]],
                        f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]"
                    )
                )
            for i in range(len(val_dict["uid"])):
                user_message = f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> {self.item} name: {val_dict['title'][i]} <EXPLAIN_POS>"
                val_input.append(
                    (
                        self.user_emb[val_dict["uid"][i]],
                        self.item_emb[val_dict["iid"][i]],
                        f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]",
                        val_dict['explanation'][i],
                    )
                )
            for i in range(len(tst_dict["uid"])):
                user_message = f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> {self.item} name: {tst_dict['title'][i]} <EXPLAIN_POS>"
                tst_input.append(
                    (
                        self.user_emb[tst_dict["uid"][i]],
                        self.item_emb[tst_dict["iid"][i]],
                        f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]",
                        tst_dict["explanation"][i],
                    )
                )


        else:    
            for i in range(len(trn_dict["uid"])):
                user_message = f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> {self.item} name: {trn_dict['title'][i]} user profile: {trn_dict['user_summary'][i]} {self.item} profile: {trn_dict['item_summary'][i]} <EXPLAIN_POS> {trn_dict['explanation'][i]}"
                trn_input.append(
                    (
                        self.user_emb[trn_dict["uid"][i]],
                        self.item_emb[trn_dict["iid"][i]],
                        f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]"
                    )
                )
            for i in range(len(val_dict["uid"])):
                user_message = f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> {self.item} name: {val_dict['title'][i]} user profile: {val_dict['user_summary'][i]} {self.item} profile: {val_dict['item_summary'][i]} <EXPLAIN_POS>"
                val_input.append(
                    (
                        self.user_emb[val_dict["uid"][i]],
                        self.item_emb[val_dict["iid"][i]],
                        f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]",
                        val_dict['explanation'][i],
                    )
                )
            for i in range(len(tst_dict["uid"])):
                user_message = f"user record: <USER_EMBED> {self.item} record: <ITEM_EMBED> {self.item} name: {tst_dict['title'][i]} user profile: {tst_dict['user_summary'][i]} {self.item} profile: {tst_dict['item_summary'][i]} <EXPLAIN_POS>"
                tst_input.append(
                    (
                        self.user_emb[tst_dict["uid"][i]],
                        self.item_emb[tst_dict["iid"][i]],
                        f"<s>[INST] <<SYS>>{self.system_prompt}<</SYS>>{user_message}[/INST]",
                        tst_dict["explanation"][i],
                    )
                )

        # Code to subsample: change 0.2 to wanted ratio to keep
        # percent_samples = int(0.2 * len(trn_input))
        # trn_input = trn_input[:percent_samples]

        # load training batch
        trn_dataset = TextDataset(trn_input)
        trn_loader = DataLoader(trn_dataset, batch_size=args.batch_size, shuffle=True)

        # load validation batch
        val_dataset = TextDataset(val_input)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

        # load testing batch
        tst_dataset = TextDataset(tst_input)
        tst_loader = DataLoader(tst_dataset, batch_size=args.batch_size, shuffle=True)

        return trn_loader, val_loader, tst_loader