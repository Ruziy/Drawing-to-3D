import argparse
from pathlib import Path

from .drawing_analyzer import analyze_pdf
from .pdf_reader import inspect_pdf
from .storage import atomic_json
from .ollama_client import LLMSettings, OllamaClient


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind",choices=["inspect","analyze"])
    parser.add_argument("--input",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--id",default="test")
    parser.add_argument("--filename",default="drawing.pdf")
    parser.add_argument("--sha256",default="")
    parser.add_argument('--mode',choices=['classic','hybrid'],default=None)
    parser.add_argument('--model',default=None,help='Точное имя установленной локальной vision-модели Ollama')
    return parser


def main():
    args = argument_parser().parse_args()
    if args.kind == "inspect":
        atomic_json(args.output/"inspection.json",inspect_pdf(args.input,args.output))
    else:
        settings=LLMSettings.from_env(args.mode,args.model)
        settings.validate_local()
        if settings.analysis_mode=='hybrid' and settings.enabled:
            status=OllamaClient(settings).status()
            atomic_json(args.output/'ollama-selection.json',{'requested_model':settings.model,'actual_model':None,
                        'status':'validated' if status['available'] else 'unavailable','availability':status})
            if not status['available']:raise ValueError(status['reason'])
        analyze_pdf(args.input,args.output,args.id,args.filename,args.sha256,settings.analysis_mode,settings.model)


if __name__ == "__main__":
    main()
