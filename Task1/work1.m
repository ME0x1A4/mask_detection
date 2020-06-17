
% %use train and validation set from dataset
% TrainFolder = "D:\Workspace\Uni\NTU\AI\Project\Task1\Data\598475_1075843_bundle_archive\Mask_Datasets\Train";
% train_set = imageDatastore(TrainFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);
% 
% ValFolder = "D:\Workspace\Uni\NTU\AI\Project\Task1\Data\598475_1075843_bundle_archive\Mask_Datasets\Validation";
% val_set = imageDatastore(ValFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);





%custom split of dataset

FullFolder = "D:\Workspace\Uni\NTU\AI\Project\Task1\Data\598475_1075843_bundle_archive\Mask_Datasets\Full_Set";
full_set = imageDatastore(FullFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);

% Adjust subsets to same size

tbl = countEachLabel(full_set)
% Determine the smallest amount of images in a category
minSetCount = min(tbl{:,2}); 

% Limit the number of images to reduce the time it takes
% run this example.
maxNumImages = 100;
minSetCount = min(maxNumImages,minSetCount);

% Use splitEachLabel method to trim the set.
full_set = splitEachLabel(full_set, minSetCount, 'randomize');

% Notice that each set now has exactly the same number of images.
countEachLabel(full_set)

%split
[train_set, val_set] = splitEachLabel(full_set, 0.3, 'randomize');





% Load pretrained network
net = resnet50();
% Inspect the first layer
net.Layers(1)
% Inspect the last layer
net.Layers(end)