import torch

from dosa.models import build_preprocessor, build_dosa, build_postprocess
from dosa.models.preprocess import Samples


def init_model(args):
    device = args.device

    preprocessor = build_preprocessor(args)
    model = build_dosa(args)
    model.to(device)
    model.eval()
    checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])

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
    :return: results
    """
    preprocessed = Samples.collate(
        [preprocessor.process(sample) for sample in samples]
    )

    with torch.no_grad():
        outputs = model(preprocessed)

    return postprocessor(outputs)
