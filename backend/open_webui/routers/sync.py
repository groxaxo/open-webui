"""
LAN Synchronization Router

Provides endpoints for syncing custom LLM configurations, tools, functions, and prompts
between Open WebUI instances on the same local network.

Security: LAN-only with token authentication
"""

import logging
import time
import aiohttp
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from open_webui.utils.auth import get_admin_user
from open_webui.models.models import Models, ModelModel
from open_webui.models.tools import Tools
from open_webui.models.functions import Functions
from open_webui.models.prompts import Prompts
from open_webui.env import SRC_LOG_LEVELS

router = APIRouter()

log = logging.getLogger(__name__)
log.setLevel(SRC_LOG_LEVELS["MAIN"])


############################
# Sync Configuration Models
############################


class SyncInstanceConfig(BaseModel):
    """Configuration for a sync instance"""
    url: str = Field(..., description="URL of the Open WebUI instance (e.g., http://192.168.1.50:3000)")
    token: str = Field(..., description="Authentication token for the instance")
    enabled: bool = Field(default=True, description="Whether syncing is enabled for this instance")


class SyncSettings(BaseModel):
    """Global sync settings"""
    auto_sync_enabled: bool = Field(default=False, description="Enable automatic syncing")
    sync_interval_minutes: int = Field(default=30, description="Sync interval in minutes")
    conflict_resolution: str = Field(
        default="newer_wins",
        description="Conflict resolution strategy: 'newer_wins', 'manual', 'local_wins', 'remote_wins'"
    )
    instances: List[SyncInstanceConfig] = Field(default=[], description="List of instances to sync with")


class SyncDataRequest(BaseModel):
    """Data sent for synchronization"""
    models: List[dict]
    tools: List[dict]
    functions: List[dict]
    prompts: List[dict]
    timestamp: int


class SyncDataResponse(BaseModel):
    """Response from sync endpoint"""
    success: bool
    models_synced: int
    tools_synced: int
    functions_synced: int
    prompts_synced: int
    conflicts: List[dict] = []
    errors: List[str] = []


############################
# Sync Endpoints
############################


@router.post("/receive", response_model=SyncDataResponse)
async def receive_sync_data(
    data: SyncDataRequest,
    user=Depends(get_admin_user)
):
    """
    Receive sync data from another Open WebUI instance.
    Applies the newer timestamp wins conflict resolution strategy by default.
    """
    try:
        results = {
            "success": True,
            "models_synced": 0,
            "tools_synced": 0,
            "functions_synced": 0,
            "prompts_synced": 0,
            "conflicts": [],
            "errors": []
        }
        
        # Sync models
        from open_webui.models.models import ModelForm
        for model_data in data.models:
            try:
                model_id = model_data.get("id")
                existing_model = Models.get_model_by_id(model_id)
                
                # Conflict resolution: newer timestamp wins
                if existing_model:
                    if model_data.get("updated_at", 0) > existing_model.updated_at:
                        model_form = ModelForm(**model_data)
                        Models.update_model_by_id(model_id, model_form)
                        results["models_synced"] += 1
                    else:
                        results["conflicts"].append({
                            "type": "model",
                            "id": model_id,
                            "resolution": "kept_local"
                        })
                else:
                    model_form = ModelForm(**model_data)
                    Models.insert_new_model(model_form, user.id)
                    results["models_synced"] += 1
            except Exception as e:
                results["errors"].append(f"Model {model_data.get('id', 'unknown')}: {str(e)}")
        
        # Sync tools
        from open_webui.models.tools import ToolForm
        for tool_data in data.tools:
            try:
                tool_id = tool_data.get("id")
                existing_tool = Tools.get_tool_by_id(tool_id)
                
                if existing_tool:
                    if tool_data.get("updated_at", 0) > existing_tool.updated_at:
                        tool_form = ToolForm(**tool_data)
                        Tools.update_tool_by_id(tool_id, tool_form)
                        results["tools_synced"] += 1
                    else:
                        results["conflicts"].append({
                            "type": "tool",
                            "id": tool_id,
                            "resolution": "kept_local"
                        })
                else:
                    tool_form = ToolForm(**tool_data)
                    Tools.insert_new_tool(user.id, tool_form)
                    results["tools_synced"] += 1
            except Exception as e:
                results["errors"].append(f"Tool {tool_data.get('id', 'unknown')}: {str(e)}")
        
        # Sync functions
        from open_webui.models.functions import FunctionForm
        for func_data in data.functions:
            try:
                func_id = func_data.get("id")
                existing_func = Functions.get_function_by_id(func_id)
                
                if existing_func:
                    if func_data.get("updated_at", 0) > existing_func.updated_at:
                        func_form = FunctionForm(**func_data)
                        Functions.update_function_by_id(func_id, func_form)
                        results["functions_synced"] += 1
                    else:
                        results["conflicts"].append({
                            "type": "function",
                            "id": func_id,
                            "resolution": "kept_local"
                        })
                else:
                    func_form = FunctionForm(**func_data)
                    Functions.insert_new_function(user.id, func_form)
                    results["functions_synced"] += 1
            except Exception as e:
                results["errors"].append(f"Function {func_data.get('id', 'unknown')}: {str(e)}")
        
        # Sync prompts
        from open_webui.models.prompts import PromptForm
        for prompt_data in data.prompts:
            try:
                prompt_command = prompt_data.get("command")
                existing_prompt = Prompts.get_prompt_by_command(prompt_command)
                
                if existing_prompt:
                    if prompt_data.get("updated_at", 0) > existing_prompt.updated_at:
                        prompt_form = PromptForm(**prompt_data)
                        Prompts.update_prompt_by_command(prompt_command, prompt_form)
                        results["prompts_synced"] += 1
                    else:
                        results["conflicts"].append({
                            "type": "prompt",
                            "id": prompt_command,
                            "resolution": "kept_local"
                        })
                else:
                    prompt_form = PromptForm(**prompt_data)
                    Prompts.insert_new_prompt(user.id, prompt_form)
                    results["prompts_synced"] += 1
            except Exception as e:
                results["errors"].append(f"Prompt {prompt_data.get('command', 'unknown')}: {str(e)}")
        
        return results
        
    except Exception as e:
        log.exception(f"Error receiving sync data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/data", response_model=SyncDataRequest)
async def get_sync_data(user=Depends(get_admin_user)):
    """
    Get current sync data to send to other instances.
    Only includes custom models (with base_model_id set).
    """
    try:
        # Only sync custom models, not base models
        custom_models = [
            model.model_dump() 
            for model in Models.get_models()
        ]
        
        sync_data = {
            "models": custom_models,
            "tools": [tool.model_dump() for tool in Tools.get_tools()],
            "functions": [func.model_dump() for func in Functions.get_functions()],
            "prompts": [prompt.model_dump() for prompt in Prompts.get_prompts()],
            "timestamp": int(time.time())
        }
        
        return sync_data
        
    except Exception as e:
        log.exception(f"Error getting sync data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/push")
async def push_to_instance(
    instance_url: str,
    instance_token: str,
    user=Depends(get_admin_user)
):
    """
    Push current sync data to another Open WebUI instance.
    """
    try:
        # Validate LAN address (basic check)
        if not (instance_url.startswith("http://192.168.") or 
                instance_url.startswith("http://10.") or 
                instance_url.startswith("http://172.") or
                instance_url.startswith("http://localhost") or
                instance_url.startswith("http://127.0.0.1")):
            raise HTTPException(
                status_code=400,
                detail="Only LAN addresses are allowed for sync (192.168.x.x, 10.x.x.x, 172.x.x.x, localhost)"
            )
        
        # Get sync data
        sync_data_dict = await get_sync_data(user=user)
        
        # Push to remote instance
        async with aiohttp.ClientSession() as session:
            url = f"{instance_url.rstrip('/')}/api/v1/sync/receive"
            headers = {
                "Authorization": f"Bearer {instance_token}",
                "Content-Type": "application/json"
            }
            
            async with session.post(
                url,
                json=sync_data_dict,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Remote sync failed: {error_text}"
                    )
                
                result = await response.json()
                return {
                    "success": True,
                    "message": f"Successfully synced to {instance_url}",
                    "details": result
                }
                
    except aiohttp.ClientError as e:
        log.exception(f"Network error during sync push: {e}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        log.exception(f"Error pushing sync data: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pull")
async def pull_from_instance(
    instance_url: str,
    instance_token: str,
    user=Depends(get_admin_user)
):
    """
    Pull sync data from another Open WebUI instance.
    """
    try:
        # Validate LAN address (basic check)
        if not (instance_url.startswith("http://192.168.") or 
                instance_url.startswith("http://10.") or 
                instance_url.startswith("http://172.") or
                instance_url.startswith("http://localhost") or
                instance_url.startswith("http://127.0.0.1")):
            raise HTTPException(
                status_code=400,
                detail="Only LAN addresses are allowed for sync (192.168.x.x, 10.x.x.x, 172.x.x.x, localhost)"
            )
        
        # Fetch data from remote instance
        async with aiohttp.ClientSession() as session:
            url = f"{instance_url.rstrip('/')}/api/v1/sync/data"
            headers = {
                "Authorization": f"Bearer {instance_token}",
            }
            
            async with session.get(
                url,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise HTTPException(
                        status_code=response.status,
                        detail=f"Remote fetch failed: {error_text}"
                    )
                
                remote_data = await response.json()
                
                # Apply the data locally
                sync_request = SyncDataRequest(**remote_data)
                result = await receive_sync_data(sync_request, user=user)
                
                return {
                    "success": True,
                    "message": f"Successfully pulled from {instance_url}",
                    "details": result
                }
                
    except aiohttp.ClientError as e:
        log.exception(f"Network error during sync pull: {e}")
        raise HTTPException(status_code=500, detail=f"Network error: {str(e)}")
    except Exception as e:
        log.exception(f"Error pulling sync data: {e}")
        raise HTTPException(status_code=500, detail=str(e))
