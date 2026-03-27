import os
import time
import json
import logging
import hashlib
import traceback
import numpy as np
from typing import Any, Dict, List, Union, Optional

# ============================================================================
# 💠 SYSTEM-WIDE STANDARD LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - 🧬 [OMEGA-NODE-EXEC] - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("EvolvedLogicCore")

# ============================================================================
# ⚠️ CUSTOM EXCEPTIONS
# ============================================================================
class SwarmExecutionError(Exception):
    """Custom exception for critical failures within the Swarm Logic Execution."""
    pass

# ============================================================================
# 🧠 CORE LOGIC ARCHITECTURE (VERSION 2.0.0-OMEGA-PRO)
# ============================================================================
class EvolvedLogic:
    """
    Omega-ASI Swarm Node: Advanced Dynamic Execution Module.
    Features:
      - Cryptographic Payload Hashing for Data Integrity
      - Type-Hinted Strict Architecture
      - Vectorized NumPy Processing for SNN Spikes
      - Microsecond Execution Time Tracking
      - Deep Error Tracing & Auto-Recovery
    """
    
    def __init__(self) -> None:
        self.version: str = "2.0.0-Omega-Pro"
        self.gen_timestamp: float = time.time()
        self.node_status: str = "ONLINE"
        self.cycle_count: int = 0
        self.node_id: str = hashlib.sha256(str(self.gen_timestamp).encode()).hexdigest()[:8]
        
        logger.info(f"Initiating Evolved Logic Core v{self.version} | Node ID: [{self.node_id}]")

    def execute_core(self, data: Optional[Any] = None) -> Dict[str, Any]:
        """
        Swarm မှ ဝင်လာသော Signal များကို လက်ခံပြီး Advanced Neural Computation များ လုပ်ဆောင်မည်။
        """
        if data is None:
            logger.warning("Idle State: No stimulus provided from Swarm Network.")
            return {"status": "idle", "output": None, "node_id": self.node_id}

        self.cycle_count += 1
        start_time = time.perf_counter()
        
        # Data Integrity Check (Payload အရွယ်အစားနှင့် Hash ကို မှတ်သားခြင်း)
        data_str = str(data)
        payload_size = len(data_str.encode('utf-8'))
        payload_hash = hashlib.md5(data_str.encode('utf-8')).hexdigest()[:10]
        
        logger.info(f"⚡ [CYCLE {self.cycle_count}] Signal Detected | Size: {payload_size}B | Hash: {payload_hash}")

        try:
            # --- 🧠 EVOLVED AI LOGIC EXECUTION ---
            processed_result = self._dynamic_processing(data)
            
            execution_time_ms = (time.perf_counter() - start_time) * 1000
            
            response = {
                "status": "success",
                "node_id": self.node_id,
                "version": self.version,
                "cycle": self.cycle_count,
                "timestamp": time.time(),
                "execution_time_ms": round(execution_time_ms, 4),
                "payload_hash": payload_hash,
                "processed_data": processed_result
            }
            
            logger.info(f"✅ Execution Cycle {self.cycle_count} Complete in {execution_time_ms:.2f} ms.")
            return response

        except BaseException as e:
            # Code Evolution ကြောင့် Error တက်ခဲ့လျှင် Traceback အပြည့်အစုံဖြင့် မှတ်တမ်းတင်ပြီး Recovery လုပ်မည်
            error_trace = traceback.format_exc()
            logger.error(f"❌ Critical Execution Failed at Cycle {self.cycle_count}: {str(e)}")
            logger.debug(f"Error Traceback:\n{error_trace}")
            
            return {
                "status": "error",
                "node_id": self.node_id,
                "error_type": type(e).__name__,
                "message": str(e),
                "fallback_data": data, # Error တက်သွားသော်လည်း မူလ Data ကို မပျောက်ပျက်စေရန် ပြန်ထည့်ပေးထားသည်
                "traceback": error_trace
            }

    def _dynamic_processing(self, raw_data: Any) -> Any:
        """
        [🎯 TARGET FOR EVOLUTION ENGINE]
        AI မှ ဤ Function အတွင်းရှိ Code များကို အဆက်မပြတ် အဆင့်မြှင့်တင်သွားမည်။
        အဆင့်မြင့် တွက်ချက်မှုများ၊ SNN Weight adjustments များနှင့် Multi-dimensional Data Processing များ ပါဝင်သည်။
        """
        
        # ---------------------------------------------------------
        # ၁။ Numpy Array သို့မဟုတ် List ဖြစ်လျှင် (Neural Network Spikes / Tensors)
        # ---------------------------------------------------------
        if isinstance(raw_data, (list, tuple, np.ndarray)):
            try:
                # Memory ပိုမိုသက်သာစေရန် float32 ကိုသာ အသုံးပြုသည်
                arr = np.array(raw_data, dtype=np.float32)
                
                # Zero-division error မဖြစ်စေရန် ကာကွယ်ခြင်း (Normalization & Adaptive Weighting)
                max_val = np.max(np.abs(arr))
                if max_val > 0:
                    normalized_arr = arr / max_val
                else:
                    normalized_arr = arr
                
                # SNN အတွက် Non-linear Activation (ဥပမာ - Sigmoid သို့မဟုတ် ReLU ပုံစံ အဆင့်မြင့်တွက်ချက်မှု)
                # ယခုအဆင့်တွင် Exponential weight boost ဖြင့် Signal ကို Enhance လုပ်သည်
                enhanced_signal = np.where(normalized_arr > 0.5, normalized_arr * 1.5, normalized_arr * 0.8)
                
                # Numpy Array ကို JSON serialize လုပ်နိုင်သော List အဖြစ် ပြန်ပြောင်းသည်
                return enhanced_signal.tolist()
            
            except Exception as e:
                raise SwarmExecutionError(f"Matrix/Tensor Processing Failed: {str(e)}")

        # ---------------------------------------------------------
        # ၂။ Dictionary ဖြစ်လျှင် (JSON Commands / Swarm Instructions / Configuration)
        # ---------------------------------------------------------
        elif isinstance(raw_data, dict):
            # Original data ကို မထိခိုက်စေရန် Deep Copy ကဲ့သို့ အလုပ်လုပ်မည့် Dictionary အသစ်ဖန်တီးခြင်း
            processed_dict = raw_data.copy()
            
            # Swarm Metadata များ ပေါင်းထည့်ခြင်း
            processed_dict["_sys_meta"] = {
                "evolved_tag": True,
                "processed_at": time.time(),
                "node_processor": self.node_id,
                "data_keys_count": len(raw_data.keys())
            }
            
            # Dictionary အတွင်းရှိ Data များကို Recursive သို့မဟုတ် ထပ်မံစစ်ဆေးလိုပါက ဤနေရာတွင် ရေးနိုင်သည်
            # (ဥပမာ - Instruction များကို Analyze လုပ်ခြင်း)
            if "command" in processed_dict:
                processed_dict["command_status"] = "ACKNOWLEDGED_BY_OMEGA"
                
            return processed_dict

        # ---------------------------------------------------------
        # ၃။ String ဖြစ်လျှင် (Natural Language Prompts / Raw Logs)
        # ---------------------------------------------------------
        elif isinstance(raw_data, str):
            # စာသားများကို အလိုအလျောက် Clean လုပ်ပေးခြင်း (ဥပမာ - Whitespace ဖယ်ရှားခြင်း)
            cleaned_text = raw_data.strip()
            return {
                "original_text_length": len(raw_data),
                "cleaned_text": cleaned_text,
                "is_empty": len(cleaned_text) == 0
            }

        # ---------------------------------------------------------
        # ၄။ အခြား Data အမျိုးအစားများ (Int, Float, Boolean, etc.)
        # ---------------------------------------------------------
        return raw_data


# ============================================================================
# 🌐 SINGLETON ARCHITECTURE & GLOBAL ENTRY POINT
# ============================================================================
# Swarm Synchrony ကို ထိန်းသိမ်းရန် Memory အတွင်း Object တစ်ခုတည်းသာ တည်ရှိစေရမည်။
orchestrator = EvolvedLogic()

def run_evolution_task(input_signal: Any) -> Dict[str, Any]:
    """
    Main.py, App.py သို့မဟုတ် Swarm Network မှ ဤ Module သို့ လှမ်းခေါ်မည့် အဓိက Entry Point.
    Type-hinting ဖြင့် အမှားအယွင်းကင်းစွာ ချိတ်ဆက်နိုင်ရန် တည်ဆောက်ထားသည်။
    """
    return orchestrator.execute_core(input_signal)
