"""
Hydronew AI Pipeline Orchestrator
==================================

Usage:
    # Train a new model (preprocess + train)
    python run_pipeline.py train

    # Run real-time classification service
    python run_pipeline.py classify

    # Just preprocess data (no training)
    python run_pipeline.py preprocess

    # Full pipeline: preprocess + train + start classification
    python run_pipeline.py all
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data_preprocessing import preprocess_data
from src.train_model import train_model
from src.data_classification import WaterQualityClassifier
from src.utils import load_config, setup_logging


def run_preprocessing():
    """
    Step 1: Preprocess raw data
    - Load from: data/raw/water_data_quality.csv
    - Validate data
    - Engineer features
    - Split train/test
    - Save to: data/processed/
    """
    print("="*60)
    print("STEP 1: DATA PREPROCESSING")
    print("="*60)
    
    config = load_config()
    setup_logging(
        config["logging"]["preprocessing_log"],
        level=config["logging"].get("level", "INFO")
    )
    
    print("\n📊 Loading and preprocessing data...")
    train_path, test_path = preprocess_data(config)
    
    print(f"\nPreprocessing complete!")
    print(f"   Training data: {train_path}")
    print(f"   Test data: {test_path}")
    print("="*60)
    
    return config


def run_training(config=None):
    """
    Step 2: Train the model
    - Load processed data
    - Apply SMOTE
    - Train RandomForest
    - Evaluate & save model
    - Save metrics
    """
    print("\n" + "="*60)
    print("STEP 2: MODEL TRAINING")
    print("="*60)
    
    if config is None:
        config = load_config()
    
    setup_logging(
        config["logging"]["training_log"],
        level=config["logging"].get("level", "INFO")
    )
    
    print("\n🤖 Training model...")
    train_model(config)
    
    print(f"\n Training complete!")
    print(f"   Model saved: models/random_forest.pkl")
    print("="*60)


def run_classification():
    """
    Step 3: Start real-time classification service
    - Load trained model
    - Connect to MQTT broker
    - Listen for messages
    - Classify water quality
    - Publish results
    """
    print("\n" + "="*60)
    print("REAL-TIME CLASSIFICATION SERVICE")
    print("="*60)
    
    # Check if model exists
    model_path = Path("models/random_forest.pkl")
    if not model_path.exists():
        print("\n Error: Model not found!")
        print(f"   Expected at: {model_path}")
        print("\n Run training first:")
        print("   python run_pipeline.py train")
        return
    
    print("\n Starting classification service...")
    print("   This will run until you press Ctrl+C")
    print()
    
    classifier = WaterQualityClassifier()
    classifier.start()


def run_full_pipeline():
    """
    Complete pipeline: Preprocess → Train → Classify
    """
    print("\n" + "="*60)
    print("🚀 FULL PIPELINE EXECUTION")
    print("="*60)
    
    # Step 1: Preprocess
    config = run_preprocessing()
    
    # Step 2: Train
    run_training(config)
    
    # Step 3: Ask if user wants to start classification
    print("\n" + "="*60)
    print("Training complete! Start classification service?")
    print("="*60)
    response = input("Start classification now? (y/n): ").lower()
    
    if response == 'y':
        run_classification()
    else:
        print("\n Pipeline complete!")
        print("\n To start classification later:")
        print("   python run_pipeline.py classify")


def main():
    parser = argparse.ArgumentParser(
        description="Hydronew AI Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train new model (preprocess + train)
  python run_pipeline.py train

  # Run classification service
  python run_pipeline.py classify

  # Just preprocess data
  python run_pipeline.py preprocess

  # Full pipeline
  python run_pipeline.py all
        """
    )
    
    parser.add_argument(
        'command',
        choices=['preprocess', 'train', 'classify', 'all'],
        help='Command to execute'
    )
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    try:
        if args.command == 'preprocess':
            run_preprocessing()
            
        elif args.command == 'train':
            # Train includes preprocessing
            print("\n Training pipeline: Preprocess → Train")
            config = run_preprocessing()
            run_training(config)
            
        elif args.command == 'classify':
            run_classification()
            
        elif args.command == 'all':
            run_full_pipeline()
            
    except KeyboardInterrupt:
        print("\n\n  Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n\n Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
