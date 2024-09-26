from collections import defaultdict

import torch

from models import build_preprocessor, build_dosa, build_postprocess


def init_model(args):
    device = args.device

    preprocessor = build_preprocessor(args)
    model = build_dosa(args)
    model.to(device)
    model.eval()
    checkpoint = torch.load(args.resume, map_location="cpu")
    # TODO, need to figure how to load with dataparallel trained
    model.load_state_dict(checkpoint)

    postprocessor = build_postprocess(args)
    postprocessor.to(device)

    return preprocessor, model, postprocessor


def infer(preprocessor, model, postprocessor, samples):
    """
    Process document given all utilities.
    :param preprocessor: preprocess features from multi-feature extractor
    :param model: model
    :param postprocessor: generate output from relation logits
    :param samples: a list of samples, each sample is the input for
        preprocessor, including a list of pages, each page is a dict containing:
        image: the image containing page objects
        boxes: bounding boxes tensor per page
        labels: labels tensor per page
        contents: content tensor corresponding to each page object
        fonts: fonts tensor for each page object per page
    :return: results
    """
    preprocessed = defaultdict(list)
    for sample in samples:
        output = preprocessor(sample)
        for key, value in output.items():
            preprocessed[key].append(value)

    batched = {}
    for key, values in preprocessed.items():
        batched[key] = torch.stack(values)

    logits = model(batched)

    return postprocessor(logits, batched["masks"])
