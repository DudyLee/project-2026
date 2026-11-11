"""Versioned PVSD prediction interface. Default 0.2; explicit 0.1 rollback."""
from pathlib import Path
import argparse
import json
import sys
from pvsd_trial_v0_2 import (ANGLES, RAW_FEATURES, INPUT_FEATURES, ROOM_FEATURES, TARGETS,
    FRACTIONS, IDENTIFIERS, encode_inputs, predict_target, sha256, read_conditions)

ROOT = Path(__file__).resolve().parent
VERSION = "0.2.0"
DEFAULT_MODEL = ROOT / "models/trial_v0_2"

class TrialPredictor:
    def __init__(self, model_dir=None, *, version=None, bim_path=None):
        versions={"0.1":"0.1.0", "0.1.0":"0.1.0", "0.2":"0.2.0", "0.2.0":"0.2.0"}
        if version is not None and version not in versions:
            raise ValueError("Supported versions: 0.1, 0.2")
        requested=versions.get(version)
        if model_dir is None:
            chosen=requested or VERSION
            model_dir=ROOT / ("models/trial_v0_1" if chosen=="0.1.0" else "models/trial_v0_2")
        else:
            chosen=json.loads((Path(model_dir)/"manifest.json").read_text(encoding="utf-8"))["version"]
            if requested is not None and requested!=chosen:
                raise ValueError("Requested version does not match model directory")
        if chosen=="0.1.0":
            if bim_path is not None:
                raise ValueError("0.1 preserves its original full-input interface; BIM lookup requires 0.2")
            from pvsd_trial_v0_1 import TrialPredictor as Legacy
            self._predictor=Legacy(model_dir)
        elif chosen=="0.2.0":
            from pvsd_trial_v0_2 import TrialPredictor as Current
            self._predictor=Current(model_dir,bim_path=bim_path)
        else:
            raise ValueError("Unsupported model version")
        self.version=chosen

    def predict(self, records, batch_size=5000):
        return self._predictor.predict(records,batch_size=batch_size)

    def __getattr__(self, name):
        return getattr(self._predictor,name)


def main(argv=None):
    parser=argparse.ArgumentParser(description="PVSD: one room/hour -> four angles and five outputs; default trial 0.2")
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    parser.add_argument("--version",choices=["0.1","0.1.0","0.2","0.2.0"])
    parser.add_argument("--model-dir",type=Path)
    parser.add_argument("--bim",type=Path,help="Optional Room_ID lookup; weather/time must be supplied")
    parser.add_argument("--batch-size",type=int,default=5000)
    parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args(argv)
    try:
        if args.input.resolve()==args.output.resolve():
            raise ValueError("Input and output must be different files")
        if args.bim and args.bim.resolve()==args.output.resolve():
            raise ValueError("BIM input and output must be different files")
        if args.output.suffix.lower() not in [".csv",".json"]:
            raise ValueError("Output must be CSV or JSON")
        if args.output.exists() and not args.overwrite:
            raise ValueError("Output exists; choose a new path or use --overwrite")
        model=TrialPredictor(args.model_dir,version=args.version,bim_path=args.bim)
        result=model.predict(read_conditions(args.input),batch_size=args.batch_size)
        args.output.parent.mkdir(parents=True,exist_ok=True)
        if args.output.suffix.lower()==".csv":
            result.to_csv(args.output,index=False,encoding="utf-8-sig")
        else:
            args.output.write_text(json.dumps(result.to_dict("records"),ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
        print(f"Trial {model.version}: {len(result)//4} input records -> {len(result)} angle rows. Saved: {args.output}")
        print("Check Coverage_Flags; flags are not confidence intervals.")
    except (ValueError,RuntimeError,OSError,KeyError) as exc:
        print(f"Input/model error: {exc}",file=sys.stderr)
        return 2
    return 0

if __name__=="__main__":
    raise SystemExit(main())
