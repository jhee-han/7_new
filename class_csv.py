import csv  # Add this import for writing to CSV

def classifier(model, data_loader, device):
    model.eval()
    acc_tracker = ratio_tracker()
    predictions = []  # To store predictions for the CSV file

    for batch_idx, item in enumerate(tqdm(data_loader)):
        model_input, categories = item
        model_input = model_input.to(device)
        original_label = torch.tensor([item.item() for item in categories], dtype=torch.int64).to(device)
        original_label = torch.tensor(original_label, dtype=torch.int64).to(device)
        answer = get_label(model, model_input, device)
        correct_num = torch.sum(answer == original_label)
        acc_tracker.update(correct_num.item(), model_input.shape[0])

        # Store predictions and labels for CSV
        for i in range(len(answer)):
            predictions.append((original_label[i].item(), answer[i].item()))

    return acc_tracker.get_ratio(), predictions  # Return accuracy and predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-b', '--batch_size', type=int,
                        default=32, help='Batch size for inference')
    parser.add_argument('-m', '--mode', type=str,
                        default='validation', help='Mode for the dataset')
    parser.add_argument('-o', '--output_csv', type=str,
                        default='test.csv', help='Output CSV file for predictions')
    
    args = parser.parse_args()
    pprint(args.__dict__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'num_workers':0, 'pin_memory':True, 'drop_last':False}

    ds_transforms = transforms.Compose([transforms.Resize((32, 32)), rescaling])
    dataloader = torch.utils.data.DataLoader(CPEN455Dataset(root_dir=args.data_dir, 
                                                            mode=args.mode, 
                                                            transform=ds_transforms), 
                                             batch_size=args.batch_size, 
                                             shuffle=True, 
                                             **kwargs)

    # Load the trained model
    model = PixelCNN(nr_resnet=1, nr_filters=40, input_channels=3, nr_logistic_mix=5,film=True)
    model = model.to(device)
    model_path = os.path.join(os.path.dirname(__file__), 'models/conditional_pixelcnn.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print('✅ model parameters loaded')
    else:
        raise FileNotFoundError(f"❌ Model file not found at {model_path}")

    model.eval()
    
    # Run classifier and get predictions
    acc, predictions = classifier(model=model, data_loader=dataloader, device=device)
    print(f"Accuracy: {acc}")

    # Save predictions to CSV
    with open(args.output_csv, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Original Label', 'Predicted Label'])  # Header
        writer.writerows(predictions)  # Write predictions
    print(f"Predictions saved to {args.output_csv}")