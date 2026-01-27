"""
Web interface for the latent-space reasoning model training.

Provides a Flask-based UI for configuring and controlling training.
"""

import os
import json
import threading
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Optional, Dict, Any

from flask import Flask, render_template, jsonify, request
from flask_socketio import SocketIO, emit

from src.config import Config


# Global state
class TrainingState:
    """Manages the global training state."""
    
    def __init__(self):
        self.is_running = False
        self.is_paused = False
        self.should_stop = False
        self.current_config: Optional[Config] = None
        self.training_thread: Optional[threading.Thread] = None
        self.metrics: Dict[str, Any] = {
            "epoch": 0,
            "total_epochs": 0,
            "step": 0,
            "total_steps": 0,
            "train_loss": 0.0,
            "avg_loss": 0.0,
            "val_loss": None,
            "best_val_loss": None,
            "status": "idle"
        }
        self.lock = threading.Lock()
        self.pause_checkpoint_path: Optional[str] = None
    
    def reset_metrics(self):
        """Reset metrics to initial state."""
        self.metrics = {
            "epoch": 0,
            "total_epochs": 0,
            "step": 0,
            "total_steps": 0,
            "train_loss": 0.0,
            "avg_loss": 0.0,
            "val_loss": None,
            "best_val_loss": None,
            "status": "idle"
        }


training_state = TrainingState()


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, 
                static_folder='static',
                template_folder='templates')
    app.config['SECRET_KEY'] = 'latent-space-training-secret'
    
    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
    
    # Store socketio reference for emitting from training thread
    app.socketio = socketio
    
    def get_default_config() -> Dict[str, Any]:
        """Get default config values from Config class."""
        default = Config()
        return asdict(default)
    
    def get_config_field_types() -> Dict[str, str]:
        """Get field types for config parameters."""
        type_map = {}
        for field in fields(Config):
            if field.type == int:
                type_map[field.name] = "int"
            elif field.type == float:
                type_map[field.name] = "float"
            elif field.type == bool:
                type_map[field.name] = "bool"
            else:
                type_map[field.name] = "str"
        return type_map
    
    @app.route('/')
    def index():
        """Serve the main page."""
        return render_template('index.html')
    
    @app.route('/api/config', methods=['GET'])
    def get_config():
        """Get default configuration values."""
        return jsonify({
            "defaults": get_default_config(),
            "types": get_config_field_types()
        })
    
    @app.route('/api/config/current', methods=['GET'])
    def get_current_config():
        """Get current training configuration (if training is active)."""
        if training_state.current_config:
            return jsonify(asdict(training_state.current_config))
        return jsonify(get_default_config())
    
    @app.route('/api/training/start', methods=['POST'])
    def start_training():
        """Start training with provided configuration."""
        with training_state.lock:
            if training_state.is_running:
                return jsonify({"error": "Training is already running"}), 400
            
            # Parse config from request
            data = request.get_json() or {}
            config = Config()
            
            # Apply overrides from request
            type_map = get_config_field_types()
            for key, value in data.items():
                if hasattr(config, key):
                    field_type = type_map.get(key, "str")
                    try:
                        if field_type == "int":
                            setattr(config, key, int(value))
                        elif field_type == "float":
                            setattr(config, key, float(value))
                        elif field_type == "bool":
                            setattr(config, key, bool(value))
                        else:
                            setattr(config, key, str(value))
                    except (ValueError, TypeError):
                        pass
            
            # Disable TUI for web training
            config.use_tui = False
            
            training_state.current_config = config
            training_state.is_running = True
            training_state.is_paused = False
            training_state.should_stop = False
            training_state.reset_metrics()
            training_state.metrics["status"] = "starting"
            training_state.metrics["total_epochs"] = config.num_epochs
            
            # Start training in background thread
            training_state.training_thread = threading.Thread(
                target=run_training_thread,
                args=(config, socketio, None),
                daemon=True
            )
            training_state.training_thread.start()
            
            return jsonify({"status": "started"})
    
    @app.route('/api/training/stop', methods=['POST'])
    def stop_training():
        """Stop training."""
        with training_state.lock:
            if not training_state.is_running:
                return jsonify({"error": "No training is running"}), 400
            
            training_state.should_stop = True
            training_state.metrics["status"] = "stopping"
            
            return jsonify({"status": "stopping"})
    
    @app.route('/api/training/pause', methods=['POST'])
    def pause_training():
        """Pause training (will save checkpoint after current step)."""
        with training_state.lock:
            if not training_state.is_running:
                return jsonify({"error": "No training is running"}), 400
            
            if training_state.is_paused:
                return jsonify({"error": "Training is already paused"}), 400
            
            training_state.is_paused = True
            training_state.metrics["status"] = "pausing"
            
            return jsonify({"status": "pausing"})
    
    @app.route('/api/training/resume', methods=['POST'])
    def resume_training():
        """Resume training from pause checkpoint."""
        with training_state.lock:
            if training_state.is_running and not training_state.is_paused:
                return jsonify({"error": "Training is already running"}), 400
            
            if not training_state.pause_checkpoint_path:
                return jsonify({"error": "No pause checkpoint available"}), 400
            
            if not Path(training_state.pause_checkpoint_path).exists():
                return jsonify({"error": "Pause checkpoint file not found"}), 400
            
            # Get config from request or use previous
            data = request.get_json() or {}
            if training_state.current_config:
                config = training_state.current_config
            else:
                config = Config()
            
            config.use_tui = False
            
            training_state.is_running = True
            training_state.is_paused = False
            training_state.should_stop = False
            training_state.metrics["status"] = "resuming"
            
            resume_path = training_state.pause_checkpoint_path
            
            # Start training in background thread with resume checkpoint
            training_state.training_thread = threading.Thread(
                target=run_training_thread,
                args=(config, socketio, resume_path),
                daemon=True
            )
            training_state.training_thread.start()
            
            return jsonify({"status": "resuming"})
    
    @app.route('/api/training/status', methods=['GET'])
    def get_training_status():
        """Get current training status and metrics."""
        return jsonify({
            "is_running": training_state.is_running,
            "is_paused": training_state.is_paused,
            "metrics": training_state.metrics,
            "has_pause_checkpoint": training_state.pause_checkpoint_path is not None
        })
    
    @socketio.on('connect')
    def handle_connect():
        """Handle client connection."""
        emit('status_update', {
            "is_running": training_state.is_running,
            "is_paused": training_state.is_paused,
            "metrics": training_state.metrics
        })
    
    @app.route('/api/inference/status', methods=['GET'])
    def get_inference_status():
        """Check if inference is available (best_model.pt exists)."""
        config = Config()
        best_model_path = Path(config.checkpoint_dir) / "best_model.pt"
        return jsonify({
            "available": best_model_path.exists(),
            "model_path": str(best_model_path) if best_model_path.exists() else None
        })
    
    @app.route('/api/inference/generate', methods=['POST'])
    def run_inference():
        """Run inference on input text."""
        data = request.get_json() or {}
        input_text = data.get("input_text", "").strip()
        
        if not input_text:
            return jsonify({"error": "No input text provided"}), 400
        
        config = Config()
        checkpoint_dir = Path(config.checkpoint_dir)
        best_model_path = checkpoint_dir / "best_model.pt"
        
        if not best_model_path.exists():
            return jsonify({"error": "No trained model found. Please train a model first."}), 400
        
        try:
            import torch
            from src.inference import LatentSpaceInference
            
            device = torch.device(config.device if torch.cuda.is_available() else "cpu")
            
            # Load best_model checkpoint
            checkpoint = torch.load(best_model_path, map_location=device)
            
            # Try to get config from best_model.pt
            if isinstance(checkpoint, dict) and "config" in checkpoint:
                config = checkpoint["config"]
            else:
                # Fallback: look for config in regular checkpoints
                checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_step_*.pt"))
                if checkpoint_files:
                    regular_checkpoint = torch.load(checkpoint_files[-1], map_location=device)
                    if isinstance(regular_checkpoint, dict) and "config" in regular_checkpoint:
                        config = regular_checkpoint["config"]
            
            # Get generation parameters from request (override config defaults)
            temperature = data.get("temperature", config.temperature)
            max_length = data.get("max_length", config.max_generation_length)
            do_sample = data.get("do_sample", config.do_sample)
            top_p = data.get("top_p", config.top_p)
            top_k = data.get("top_k", config.top_k)
            
            # Initialize inference model with the correct config
            inference_model = LatentSpaceInference(str(best_model_path), config, device)
            
            # Generate response
            output = inference_model.generate(
                input_text=input_text,
                max_length=int(max_length),
                temperature=float(temperature),
                do_sample=bool(do_sample),
                top_p=float(top_p),
                top_k=int(top_k)
            )
            
            return jsonify({
                "input": input_text,
                "output": output,
                "parameters": {
                    "temperature": temperature,
                    "max_length": max_length,
                    "do_sample": do_sample,
                    "top_p": top_p,
                    "top_k": top_k
                }
            })
            
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    
    return app, socketio


def run_training_thread(config: Config, socketio: SocketIO, resume_from: Optional[str] = None):
    """Run training in a background thread with metric updates."""
    import torch
    from torch.optim import AdamW
    from torch.cuda.amp import GradScaler, autocast
    
    from src.models import LatentSpaceModel
    from src.data import create_dataloaders
    from src.train import compute_loss, save_checkpoint
    
    def emit_metrics():
        """Emit current metrics to all connected clients."""
        socketio.emit('metrics_update', training_state.metrics)
    
    try:
        device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        training_state.metrics["status"] = "loading_data"
        emit_metrics()
        
        train_loader, val_loader = create_dataloaders(
            config, 
            train_samples=config.max_train_samples, 
            val_samples=1000
        )
        
        training_state.metrics["status"] = "loading_model"
        emit_metrics()
        
        model = LatentSpaceModel(config).to(device)
        trainable_params_list = [p for p in model.parameters() if p.requires_grad]
        optimizer = AdamW(trainable_params_list, lr=config.learning_rate, weight_decay=config.weight_decay)
        scaler = GradScaler(enabled=config.use_fp16)
        
        start_epoch = 0
        start_step = 0
        best_val_loss = float('inf')
        
        # Resume from checkpoint if provided
        if resume_from and Path(resume_from).exists():
            training_state.metrics["status"] = "loading_checkpoint"
            emit_metrics()
            
            checkpoint = torch.load(resume_from, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
            start_epoch = checkpoint.get("epoch", 0)
            start_step = checkpoint.get("step", 0)
            best_val_loss = checkpoint.get("best_val_loss", float('inf'))
            
            training_state.metrics["best_val_loss"] = best_val_loss if best_val_loss != float('inf') else None
        
        training_state.metrics["total_steps"] = len(train_loader) * config.num_epochs
        training_state.metrics["status"] = "training"
        emit_metrics()
        
        global_step = start_step
        
        for epoch in range(start_epoch, config.num_epochs):
            if training_state.should_stop:
                break
                
            model.train()
            total_loss = 0.0
            num_batches = 0
            
            training_state.metrics["epoch"] = epoch + 1
            training_state.metrics["total_epochs"] = config.num_epochs
            emit_metrics()
            
            optimizer.zero_grad()
            
            for step, batch in enumerate(train_loader):
                # Check for stop/pause
                if training_state.should_stop:
                    break
                
                if training_state.is_paused:
                    # Save pause checkpoint
                    training_state.metrics["status"] = "saving_pause_checkpoint"
                    emit_metrics()
                    
                    checkpoint_dir = Path(config.checkpoint_dir)
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    pause_path = checkpoint_dir / "temp_pause.pt"
                    
                    torch.save({
                        "step": global_step,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "best_val_loss": best_val_loss,
                        "config": config
                    }, pause_path, _use_new_zipfile_serialization=False)
                    
                    training_state.pause_checkpoint_path = str(pause_path)
                    training_state.metrics["status"] = "paused"
                    training_state.is_running = False
                    emit_metrics()
                    return
                
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target_ids = batch["target_ids"].to(device)
                target_attention_mask = batch["target_attention_mask"].to(device)
                
                with autocast(enabled=config.use_fp16):
                    logits = model(input_ids, attention_mask, target_ids, target_attention_mask)
                    loss = compute_loss(logits, target_ids, target_attention_mask)
                    loss = loss / config.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                total_loss += loss.item() * config.gradient_accumulation_steps
                num_batches += 1
                
                if (step + 1) % config.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    avg_loss = total_loss / num_batches
                    
                    training_state.metrics["step"] = global_step
                    training_state.metrics["train_loss"] = loss.item() * config.gradient_accumulation_steps
                    training_state.metrics["avg_loss"] = avg_loss
                    emit_metrics()
                    
                    if global_step % config.save_every == 0:
                        save_checkpoint(model, optimizer, scaler, global_step, epoch, config)
            
            if training_state.should_stop:
                break
            
            # Validation
            training_state.metrics["status"] = "validating"
            emit_metrics()
            
            model.eval()
            val_total_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    target_ids = batch["target_ids"].to(device)
                    target_attention_mask = batch["target_attention_mask"].to(device)
                    
                    with autocast(enabled=config.use_fp16):
                        logits = model(input_ids, attention_mask, target_ids, target_attention_mask)
                        loss = compute_loss(logits, target_ids, target_attention_mask)
                    
                    val_total_loss += loss.item()
                    val_batches += 1
            
            val_loss = val_total_loss / val_batches
            training_state.metrics["val_loss"] = val_loss
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                training_state.metrics["best_val_loss"] = best_val_loss
                
                checkpoint_dir = Path(config.checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                best_model_path = checkpoint_dir / "best_model.pt"
                torch.save(model.state_dict(), best_model_path, _use_new_zipfile_serialization=False)
            
            training_state.metrics["status"] = "training"
            emit_metrics()
        
        # Training complete
        training_state.metrics["status"] = "completed"
        emit_metrics()
        
    except Exception as e:
        training_state.metrics["status"] = f"error: {str(e)}"
        socketio.emit('metrics_update', training_state.metrics)
    
    finally:
        with training_state.lock:
            training_state.is_running = False
            training_state.is_paused = False
            training_state.should_stop = False


def run_web_server(host: str = "127.0.0.1", port: int = 5000):
    """Run the web server."""
    app, socketio = create_app()
    print(f"\n{'='*50}")
    print("Latent-Space Model - Web Training Interface")
    print(f"{'='*50}")
    print(f"\nServer running at: http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    print(f"{'='*50}\n")
    
    socketio.run(app, host=host, port=port, debug=False, allow_unsafe_werkzeug=True)
