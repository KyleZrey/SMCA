import torch
from torcheval.metrics import BinaryPrecision, BinaryRecall, BinaryF1Score

def test_model(model, dense_layer, dataloader, criterion, device):
    model.eval()
    dense_layer.eval()  # Set the model to evaluation mode
    total_loss = 0

    # Initialize the metrics for binary classification
    precision_metric = BinaryPrecision().to(device)
    recall_metric = BinaryRecall().to(device)
    f1_metric = BinaryF1Score().to(device)

    with torch.no_grad():
        for text_features, audio_features, video_features, targets in dataloader:
            text_features, audio_features, video_features, targets = (
                text_features.to(device),
                audio_features.to(device),
                video_features.to(device),
                targets.to(device).view(-1)
            )
            
            # Pass inputs through SMCA model
            outputs = model(modalityAlpha=text_features.to(device), modalityBeta=audio_features.to(device), modalityGamma=video_features.to(device), device=device)

            # Check if padding is necessary
            output_size = outputs.size(1)
            dense_input_size = dense_layer.fc.in_features
            
            if output_size < dense_input_size:
                # Pad the outputs if they are smaller than the expected size for the dense layer
                padding_size = dense_input_size - output_size
                # Pad on the second dimension (feature dimension)
                outputs = torch.nn.functional.pad(outputs, (0, padding_size))
            elif output_size > dense_input_size:
                # In case outputs are larger (though unlikely, we trim)
                outputs = outputs[:, :dense_input_size]

            # Pass the fused features through the dense layer
            predictions = torch.sigmoid(dense_layer(outputs)).view(-1)  
            
            # Compute loss
            loss = criterion(predictions, targets)
            total_loss += loss.item()

            # Apply threshold to get binary predictions
            preds = (predictions > 0.5).float()
            
            # Update the precision, recall, and F1 score metrics
            precision_metric.update(preds.long(), targets.long())
            recall_metric.update(preds.long(), targets.long())
            f1_metric.update(preds.long(), targets.long())

    # Compute precision, recall, and F1 score
    precision = precision_metric.compute().item()
    recall = recall_metric.compute().item()
    f1_score = f1_metric.compute().item()

    average_loss = total_loss / len(dataloader)

    print(f"Test Loss: {average_loss:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1_score:.4f}")

    return average_loss, precision, recall, f1_score
