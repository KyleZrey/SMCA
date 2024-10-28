import torch

def output_max(model, dataloader, device):
    max_output_size = 0

    model.eval()
    with torch.no_grad():
        for text_features, audio_features, video_features, targets in dataloader:
            # Move features to device
            text_features = text_features.to(device)
            audio_features = audio_features.to(device)
            video_features = video_features.to(device)

            # Check the shape and remove unnecessary dimensions
            if audio_features.dim() == 4:  # Replace modalityAlpha with the actual tensor variable
                audio_features = audio_features.squeeze(1)  # Remove the extra dimension if needed
            if text_features.dim() == 4:  # Replace modalityBeta with the actual tensor variable
                text_features = text_features.squeeze(1)
            if video_features.dim() == 4:  # Replace modalityGamma with the actual tensor variable
                video_features = video_features.squeeze(1)

            # Pass inputs through the SMCA model
            outputs = model(
                modalityAlpha=audio_features, 
                modalityBeta=text_features, 
                modalityGamma=video_features,
                # device=device (not needed here)
            )

            # Compare and store the maximum output size
            if outputs.size(1) > max_output_size:
                max_output_size = outputs.size(1)

    return max_output_size
