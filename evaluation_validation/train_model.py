import torch


def train_model(model, dense_layer, dataloader, criterion, optimizer, device):
    model.train()
    dense_layer.train()  # Set the model to training mode
    total_loss = 0.0

    for text_features, audio_features, video_features, targets in dataloader:
        text_features, audio_features, video_features, targets = (
            text_features.to(device),
            audio_features.to(device),
            video_features.to(device),
            targets.to(device).view(-1)
        )
        
        optimizer.zero_grad()
        
        # Pass inputs through SMCA model
        
        outputs = model(modalityAlpha=text_features, modalityBeta=audio_features, modalityGamma=video_features, device=device)
        # outputs = model(modalityAlpha=text_features, modalityBeta=video_features, modalityGamma=audio_features, device=device)
        # outputs = model(modalityAlpha=audio_features, modalityBeta=text_features, modalityGamma=video_features, device=device)
        # outputs = model(modalityAlpha=audio_features, modalityBeta=video_features, modalityGamma=text_features, device=device)
        # outputs = model(modalityAlpha=video_features, modalityBeta=audio_features, modalityGamma=text_features, device=device)
        # outputs = model(modalityAlpha=video_features, modalityBeta=text_features, modalityGamma=audio_features, device=device)
        
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
        predictions = dense_layer(outputs).view(-1)

        # Compute loss
        loss = criterion(predictions, targets)
        total_loss += loss.item()
        # Backward pass and optimization
        loss.backward()
        optimizer.step()



    return total_loss / len(dataloader)
