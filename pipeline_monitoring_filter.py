"""
title: Filter Pipeline
date: 2025-09-19
version: 1.2
description: A filter pipeline implemented using Prometheus
"""
from typing import List, Optional
import time
import os
import uuid
import json
from prometheus_client import Counter, Histogram, Gauge, start_http_server

from utils.pipelines.main import get_last_assistant_message
from pydantic import BaseModel


GLOBAL_user_count = 0
# ===================== Prometheus Metrics =====================
PIPELINE_REQUESTS = Counter(
    "pipeline_requests_total",
    "Total requests processed by the filter",
    ["stage", "task"]                       #stage-> [inlet,outlet],    task->[user_response, llm_response]
)

PIPELINE_ERRORS = Counter(
    "pipeline_errors_total",
    "Total errors encountered in pipeline",
    ["stage"]
)

PIPELINE_INLET_OUTLET_LATENCY = Histogram(
    "pipeline_request_latency_seconds_for_inlet_outlet_filter",
    "Latency of pipeline filtering stages",
    ["stage"]                               #count-> how many   each_of_bucket - <= num_le count    seconds_sum=total_latency  calculate latency of inlet and outlet functions / probably we can ignore this
)

PIPELINE__TIME_CAPTURE = Gauge(
    "pipeline_time_capture_seconds",
    "Unix timestamp of pipeline stage execution",   #pipeline processing time latency
    ["stage", "model_id"]
)

ACTIVE_SPANS = Gauge(
    "pipeline_active_spans",
    "Number of active spans currently being tracked",    #number:active spans count
    ["model_id"]
)

MODEL_USAGE = Counter(
    "pipeline_model_usage_total",
    "Number of times each model is used",
    ["model_id"]              #model_id, num of times the model is being called
)

class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        debug: bool = False
        insert_tags: bool = True

    def __init__(self):

        print("PID:", os.getpid()) # to check user scope

        self.type = "filter" # This is a filter so keep the name as filter to identification ; default is a pipeline and as there is no pipe function it gives an error
        self.name = "pipeline_monitor" # The setup name for the filter 

        self.valves = self.Valves(
            **{
                "pipelines": ["*"], # use the filter for all the pipelines , or can be customize by adding pipeline names
                "debug": os.getenv("DEBUG_MODE", "false").lower() == "true",
            }
        )
        

        
        self.chat_spans = {}
        # Dictionary to store model names for each chat
        # self.model_names = {}
        self.is_pipe = True
        self.model_id_run = None
        


    def log(self, message: str, suppress_repeats: bool = False):
        if self.valves.debug:
            print(f"[DEBUG] {message}")


    async def on_startup(self):
        self.log(f"on_startup triggered for {__name__}")
        start_http_server(9100)  # Prometheus metrics exposed at :9100/metrics

    async def on_shutdown(self):
        self.log(f"on_shutdown triggered for {__name__}")


    def _build_tags(self, task_name: str) -> list:
        tags_list = []
        if self.valves.insert_tags:
            tags_list.append("open-webui")
            if task_name not in ["user_response", "llm_response"]:
                tags_list.append(task_name)
        return tags_list



    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        start_time = time.time() # user_response time
        global GLOBAL_user_count
        GLOBAL_user_count += 1
        try:
            metadata = body.get("metadata", {})     #body message send by the user is captured by the inlte funtion
            
            pipeline_info = metadata.get("pipeline") or metadata.get("model", {}).get("pipeline", {})
            if not pipeline_info.get("type") == "pipe":     #check whether a pipeline or not, we are only monitoring the pipelines
                self.log("not a pipe-inlet bypass") # * works in llm models also that's why 
                self.is_pipe = False
                return body

            self.is_pipe = True #To notify the outlet
            self.log("Inside the --INLET--")  

            model_info = metadata.get("model", {})
            model_id = body.get("model")
            

            required_keys = ["model", "messages"]
            missing_keys = [key for key in required_keys if key not in body]
            if missing_keys:
                raise ValueError(f"Missing keys in request body: {missing_keys}")

            task_name = metadata.get("task", "user_response")

            # Update Prometheus metrics
            PIPELINE_REQUESTS.labels(stage="inlet", task=task_name).inc()
            ACTIVE_SPANS.labels(model_id=model_id).set(GLOBAL_user_count)

            duration = time.time() - start_time
            PIPELINE_INLET_OUTLET_LATENCY.labels(stage="inlet").observe(duration)

            model_id = metadata.get("model", {}).get("id")
            self.model_id_run = model_id

            # Update Prometheus metrics
            MODEL_USAGE.labels(
                    model_id=model_id
                ).inc()

            # Update Prometheus metrics
            PIPELINE__TIME_CAPTURE.labels(
                stage="inlet", model_id=model_id
            ).set(time.time())

            return body

        except Exception as e:
            PIPELINE_ERRORS.labels(stage="inlet").inc()
            raise e

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        start_time = time.time()
        
        try:
            metadata = body.get("metadata", {}) 
            if not self.is_pipe:
                self.log("not a pipe-outlet bypass")
                return body
            self.log("Inside the --OUTLET--")

            PIPELINE__TIME_CAPTURE.labels(
                stage="outlet", model_id=self.model_id_run
            ).set(start_time)
            

            task_name = body.get("metadata", {}).get("task", "llm_response")

            # Update Prometheus metrics
            PIPELINE_REQUESTS.labels(stage="outlet", task=task_name).inc()

            duration = time.time() - start_time
            PIPELINE_INLET_OUTLET_LATENCY.labels(stage="outlet").observe(duration) 
            global GLOBAL_user_count
            GLOBAL_user_count -= 1
            print(f"GLOBAL_user_count {GLOBAL_user_count} ")
            ACTIVE_SPANS.labels(model_id=self.model_id_run).set(GLOBAL_user_count)
            self.model_id_run = None
            return body

        except Exception as e:
            PIPELINE_ERRORS.labels(stage="outlet").inc()
            self.model_id_run = None
            raise e
    
