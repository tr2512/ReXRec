import os
import pickle
import json
import torch
import torch.nn as nn
from models.explainer import Explainer
from utils.data_handler import DataHandler
from utils.parse import args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device {device}")

class XRec:
    def __init__(self):
        print(f"dataset: {args.dataset}")
        self.model = Explainer().to(device)
        self.data_handler = DataHandler()

        self.trn_loader, self.val_loader, self.tst_loader = self.data_handler.load_data()

        self.user_embedding_converter_path = f"./data/{args.dataset}/user_converter_{args.out_name}.pkl"
        self.item_embedding_converter_path = f"./data/{args.dataset}/item_converter_{args.out_name}.pkl"

        self.tst_predictions_path = f"./data/{args.dataset}/tst_predictions_{args.out_name}.pkl"
        self.tst_references_path = f"./data/{args.dataset}/tst_references_{args.out_name}.pkl"

    def train(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)

        log_file_path = f"./data/{args.dataset}/logs_{args.out_name}.txt"
        with open(log_file_path, "w") as file:
            file.write("Epoch,Step,Loss\n") 
        for epoch in range(args.epochs):
            total_loss = 0
            self.model.train()
            for i, batch in enumerate(self.trn_loader):
                if not args.random_random:
                    user_embed, item_embed, input_text = batch
                else:
                    input_text = batch
                    b = len(input_text)
                    user_embed = nn.init.xavier_uniform_(torch.empty(b, 64))
                    item_embed = nn.init.xavier_uniform_(torch.empty(b, 64))
                user_embed = user_embed.to(device)
                item_embed = item_embed.to(device)
                input_ids, outputs, explain_pos_position = self.model.forward(user_embed, item_embed, input_text)
                input_ids = input_ids.to(device)
                explain_pos_position = explain_pos_position.to(device)
                optimizer.zero_grad()
                loss = self.model.loss(input_ids, outputs, explain_pos_position, device)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                if i % 100 == 0 and i != 0:
                    with open(log_file_path, "a") as file:
                        file.write(f"{epoch},{i},{loss.item()}\n")
                    torch.save(
                       {
                           "epoch": epoch,
                           "step": i,
                           "model_state_dict": self.model.user_embedding_converter.state_dict(),
                           "optimizer_state_dict": optimizer.state_dict(),
                           "loss": loss.item(),
                       },
                       self.user_embedding_converter_path,
                    )
                    torch.save(
                       {
                           "epoch": epoch,
                           "step": i,
                           "model_state_dict": self.model.item_embedding_converter.state_dict(),
                           "optimizer_state_dict": optimizer.state_dict(),
                           "loss": loss.item(),
                       },
                       self.item_embedding_converter_path,
                    )
                    print(
                        f"Epoch [{epoch}/{args.epochs}], Step [{i}/{len(self.trn_loader)}], Loss: {loss.item()}"
                    )
                    # print(f"Generated Explanation: {outputs[0]}")

            print(f"Epoch [{epoch}/{args.epochs}], Loss: {total_loss}")
            # Save the model
            torch.save(
                self.model.user_embedding_converter.state_dict(),
                self.user_embedding_converter_path,
            )
            torch.save(
                self.model.item_embedding_converter.state_dict(),
                self.item_embedding_converter_path,
            )
            print(f"Saved model to {self.user_embedding_converter_path}")
            print(f"Saved model to {self.item_embedding_converter_path}")

    def evaluate(self):
        loader = self.tst_loader
        predictions_path = self.tst_predictions_path
        references_path = self.tst_references_path

        # load model
        self.model.user_embedding_converter.load_state_dict(
            torch.load(self.user_embedding_converter_path)
        )
        self.model.item_embedding_converter.load_state_dict(
            torch.load(self.item_embedding_converter_path)
        )
        self.model.eval()
        predictions = []
        references = []
        with torch.no_grad():
            for i, batch in enumerate(loader):
                if args.random_random:
                    input_text, explain = batch
                    b = len(input_text)
                    user_embed = nn.init.xavier_uniform_(torch.empty(b, 64))
                    item_embed = nn.init.xavier_uniform_(torch.empty(b, 64))
                else:
                    user_embed, item_embed, input_text, explain = batch
                user_embed = user_embed.to(device)
                item_embed = item_embed.to(device)

                outputs = self.model.generate(user_embed, item_embed, input_text)
                end_idx = outputs[0].find("[")
                if end_idx != -1:
                    outputs[0] = outputs[0][:end_idx]

                predictions.append(outputs[0])
                references.append(explain[0])
                if i % 10 == 0 and i != 0:
                    print(f"Step [{i}/{len(loader)}]", flush=True)
                    print(f"Generated Explanation: {outputs[0]}", flush=True)

        with open(predictions_path, "wb") as file:
            pickle.dump(predictions, file)
        with open(references_path, "wb") as file:
            pickle.dump(references, file)
        print(f"Saved predictions to {predictions_path}")
        print(f"Saved references to {references_path}")   

def main():
    sample = XRec()
    if args.mode == "finetune":
        print("Finetune model...")
        sample.train()
    elif args.mode == "generate":
        print("Generating explanations...")
        sample.evaluate()

if __name__ == "__main__":
    main()