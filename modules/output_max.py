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

            # Pass inputs through the SMCA model
            outputs = model(
                modalityAlpha=audio_features, 
                modalityBeta=text_features, 
                modalityGamma=video_features,
                device=device
            )

            # Compare and store the maximum output size
            if outputs.size(1) > max_output_size:
                max_output_size = outputs.size(1)

    return max_output_size