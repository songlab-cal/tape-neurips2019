from typing import Dict, Type, List, Optional
from tensorflow.keras import Model
from rinokeras.layers import Stack
from sacred import Ingredient

from tape.task_models import FreezeWeights
from .Task import Task

from .BeplerLanguageModelingTask import BeplerLanguageModelingTask
from .BeplerContactMapTask import BeplerContactMapTask
from .ContactMapTask import ContactMapTask
from .StabilityTask import StabilityTask
from .FluorescenceTask import FluorescenceTask
from .LanguageModelingTask import LanguageModelingTask
from .MaskedLanguageModelingTask import MaskedLanguageModelingTask, mask_params
from .NetsurfTask import NetsurfTask
from .BeplerPairedScopeTask import BeplerPairedScopeTask
from .RemoteHomologyTask import RemoteHomologyTask
from .SecondaryStructureTask import SecondaryStructureTask, secondary_structure_params
from .UnidirectionalLanguageModelingTask import UnidirectionalLanguageModelingTask


class TaskBuilder:

    tasks: Dict[str, Type[Task]] = {
        'bepler_contact_map': BeplerContactMapTask,
        'bepler_language_modeling': BeplerLanguageModelingTask,
        'bepler_paired_scope': BeplerPairedScopeTask,
        'contact_map': ContactMapTask,
        'fluorescence': FluorescenceTask,
        'language_modeling': LanguageModelingTask,
        'masked_language_modeling': MaskedLanguageModelingTask,
        'netsurf': NetsurfTask,
        'remote_homology': RemoteHomologyTask,
        'secondary_structure': SecondaryStructureTask,
        'stability': StabilityTask,
        'unidirectional_language_modeling': UnidirectionalLanguageModelingTask
    }

    params: List[Ingredient] = [mask_params, secondary_structure_params]

    @staticmethod
    def build_task(task_name: str) -> Task:
        task_name = task_name.lower()
        if task_name[-5:] == '_task':
            task_name = task_name[:-5]
        elif task_name[-4:] == 'task':
            task_name = task_name[:-4]

        task = TaskBuilder.tasks[task_name]
        return task()  # type: ignore

    @staticmethod
    def build_tasks(task_names: List[str]) -> List[Task]:
        return [TaskBuilder.build_task(task_name) for task_name in task_names]

    @staticmethod
    def build_task_model(embedding_model: Model,
                         tasks: List[Task],
                         freeze_embedding_weights: bool) -> Model:
        layers = [embedding_model]

        if freeze_embedding_weights:
            layers.append(FreezeWeights())

        for task in tasks:
            layers = task.build_output_model(layers)
        return Stack(layers)

    @staticmethod
    def add_task(task_name: str, task: Type[Task], params: Optional[Ingredient] = None) -> None:
        assert isinstance(task, type)
        TaskBuilder.tasks[task_name] = task

        if params is not None:
            TaskBuilder.params.append(params)
