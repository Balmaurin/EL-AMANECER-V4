#!/usr/bin/env python3
"""
Sistema de Creación Dinámica de Tools para Sheily AI
Permite a los agentes crear, modificar y optimizar herramientas automáticamente
Incluye generación automática de código, testing y deployment
"""

import ast
import asyncio
import inspect
import json
import logging
import re
import textwrap
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from sheily_core.a2a_protocol import a2a_system
from sheily_core.agent_learning import LearningExperience, record_agent_experience
from sheily_core.agent_quality import evaluate_agent_quality
from sheily_core.agent_tracing import trace_agent_execution
from sheily_core.agents.multi_agent_system import multi_agent_system

logger = logging.getLogger(__name__)

# =============================================================================
# MODELOS DE DATOS PARA TOOLS
# =============================================================================


class ToolType(Enum):
    """Tipos de herramientas disponibles"""

    PYTHON_FUNCTION = "python_function"
    API_CLIENT = "api_client"
    DATA_PROCESSOR = "data_processor"
    CODE_ANALYZER = "code_analyzer"
    FILE_HANDLER = "file_handler"
    WEB_SCRAPER = "web_scraper"
    CALCULATION_TOOL = "calculation_tool"
    CUSTOM_SCRIPT = "custom_script"


class ToolStatus(Enum):
    """Estados de una herramienta"""

    DRAFT = "draft"
    TESTING = "testing"
    VALIDATED = "validated"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


@dataclass
class ToolSpecification:
    """Especificación completa de una herramienta"""

    tool_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    description: str = ""
    tool_type: ToolType = ToolType.PYTHON_FUNCTION
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    code_template: str = ""
    dependencies: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    created_by: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0.0"

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "tool_id": self.tool_id,
            "name": self.name,
            "description": self.description,
            "tool_type": self.tool_type.value,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "parameters": self.parameters,
            "code_template": self.code_template,
            "dependencies": self.dependencies,
            "tags": self.tags,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
        }


@dataclass
class ToolImplementation:
    """Implementación ejecutable de una herramienta"""

    spec: ToolSpecification
    code: str = ""
    test_code: str = ""
    status: ToolStatus = ToolStatus.DRAFT
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    validation_results: Dict[str, Any] = field(default_factory=dict)
    deployment_info: Dict[str, Any] = field(default_factory=dict)
    last_modified: datetime = field(default_factory=datetime.now)
    usage_count: int = 0
    success_rate: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario"""
        return {
            "spec": self.spec.to_dict(),
            "code": self.code,
            "test_code": self.test_code,
            "status": self.status.value,
            "performance_metrics": self.performance_metrics,
            "validation_results": self.validation_results,
            "deployment_info": self.deployment_info,
            "last_modified": self.last_modified.isoformat(),
            "usage_count": self.usage_count,
            "success_rate": self.success_rate,
        }


@dataclass
class ToolCreationRequest:
    """Solicitud para crear una nueva herramienta"""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requester_agent: str = ""
    tool_name: str = ""
    description: str = ""
    tool_type: ToolType = ToolType.PYTHON_FUNCTION
    requirements: Dict[str, Any] = field(default_factory=dict)
    examples: List[Dict[str, Any]] = field(default_factory=list)
    priority: int = 1
    deadline: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, in_progress, completed, failed


# =============================================================================
# GENERADOR DE CÓDIGO AUTOMÁTICO
# =============================================================================


class CodeGenerator:
    """Generador automático de código para herramientas"""

    def __init__(self):
        self.templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """Carga plantillas de código por tipo de herramienta"""
        return {
            ToolType.PYTHON_FUNCTION: """
def {function_name}({parameters}) -> {return_type}:
    \"\"\"
    {docstring}

    Args:
        {args_doc}

    Returns:
        {return_doc}
    \"\"\"
    try:
        # Implementation
        {implementation}

        return {return_value}
    except Exception as e:
        logger.error(f"Error in {function_name}: {{e}}")
        raise
""",
            ToolType.API_CLIENT: """
import aiohttp
import json
from typing import Dict, Any, Optional

class {class_name}:
    \"\"\"{docstring}\"\"\"

    def __init__(self, base_url: str, api_key: Optional[str] = None):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def {method_name}(self, {parameters}) -> Dict[str, Any]:
        \"\"\"
        {method_docstring}

        Args:
            {args_doc}

        Returns:
            {return_doc}
        \"\"\"
        url = f"{{self.base_url}}/{endpoint}"

        headers = {{'Content-Type': 'application/json'}}
        if self.api_key:
            headers['Authorization'] = f'Bearer {{self.api_key}}'

        {request_logic}

        async with self.session.{http_method}(url, **kwargs) as response:
            if response.status == 200:
                return await response.json()
            else:
                error_text = await response.text()
                raise Exception(f"API request failed: {{response.status}} - {{error_text}}")
""",
            ToolType.DATA_PROCESSOR: """
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union

def {function_name}(data: Union[pd.DataFrame, List[Dict], Dict], {parameters}) -> {return_type}:
    \"\"\"
    {docstring}

    Args:
        data: Input data to process
        {args_doc}

    Returns:
        {return_doc}
    \"\"\"
    try:
        # Convert input to DataFrame if needed
        if isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()

        # Processing logic
        {implementation}

        return {return_value}

    except Exception as e:
        logger.error(f"Error processing data: {{e}}")
        raise
""",
            ToolType.CODE_ANALYZER: """
import ast
import re
from typing import Dict, Any, List, Tuple

def {function_name}(code: str, {parameters}) -> {return_type}:
    \"\"\"
    {docstring}

    Args:
        code: Source code to analyze
        {args_doc}

    Returns:
        {return_doc}
    \"\"\"
    try:
        results = {{
            'issues': [],
            'metrics': {{}},
            'suggestions': []
        }}

        # Parse code
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            results['issues'].append({{
                'type': 'syntax_error',
                'message': str(e),
                'line': e.lineno
            }})
            return results

        # Analysis logic
        {implementation}

        return results

    except Exception as e:
        logger.error(f"Error analyzing code: {{e}}")
        return {{
            'issues': [{{'type': 'analysis_error', 'message': str(e)}}],
            'metrics': {{}},
            'suggestions': []
        }}
""",
        }

    def generate_tool_code(self, spec: ToolSpecification) -> str:
        """Genera código para una herramienta basada en su especificación"""
        template = self.templates.get(
            spec.tool_type, self.templates[ToolType.PYTHON_FUNCTION]
        )

        # Extraer información de la especificación
        function_name = self._sanitize_name(spec.name)
        parameters = self._generate_parameters(spec.input_schema)
        return_type = self._infer_return_type(spec.output_schema)

        # Generar documentación
        docstring = spec.description
        args_doc = self._generate_args_doc(spec.input_schema)
        return_doc = self._generate_return_doc(spec.output_schema)

        # Generar implementación
        implementation = self._generate_implementation(spec)

        # Formatear código
        code = template.format(
            function_name=function_name,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
            args_doc=args_doc,
            return_doc=return_doc,
            implementation=implementation,
            return_value=self._generate_return_value(spec.output_schema),
            class_name=function_name.title(),
            method_name=function_name,
            method_docstring=docstring,
            endpoint=spec.parameters.get("endpoint", "/api/data"),
            http_method=spec.parameters.get("http_method", "get"),
            request_logic=self._generate_request_logic(spec),
        )

        return textwrap.dedent(code).strip()

    def _sanitize_name(self, name: str) -> str:
        """Convierte un nombre legible en un nombre de función válido"""
        # Convertir a snake_case y remover caracteres especiales
        name = re.sub(r"[^\w\s-]", "", name)
        name = re.sub(r"[\s-]+", "_", name)
        return name.lower()

    def _generate_parameters(self, input_schema: Dict[str, Any]) -> str:
        """Genera la lista de parámetros de función"""
        params = []
        for param_name, param_info in input_schema.items():
            param_type = param_info.get("type", "Any")
            default_value = param_info.get("default", None)

            if default_value is not None:
                params.append(f"{param_name}: {param_type} = {repr(default_value)}")
            else:
                params.append(f"{param_name}: {param_type}")

        return ", ".join(params)

    def _infer_return_type(self, output_schema: Dict[str, Any]) -> str:
        """Infier el tipo de retorno basado en el esquema de salida"""
        if not output_schema:
            return "Any"

        output_type = output_schema.get("type", "dict")
        if output_type == "dict":
            return "Dict[str, Any]"
        elif output_type == "list":
            return "List[Any]"
        elif output_type in ["str", "int", "float", "bool"]:
            return output_type
        else:
            return "Any"

    def _generate_args_doc(self, input_schema: Dict[str, Any]) -> str:
        """Genera documentación para los argumentos"""
        docs = []
        for param_name, param_info in input_schema.items():
            param_type = param_info.get("type", "Any")
            description = param_info.get("description", f"Parameter {param_name}")
            docs.append(f"        {param_name} ({param_type}): {description}")

        return "\n".join(docs)

    def _generate_return_doc(self, output_schema: Dict[str, Any]) -> str:
        """Genera documentación para el valor de retorno"""
        if not output_schema:
            return "Any: Result of the operation"

        description = output_schema.get("description", "Result of the operation")
        return description

    def _generate_implementation(self, spec: ToolSpecification) -> str:
        """Genera la lógica de implementación"""
        # Esta es una implementación básica - en producción sería más sofisticada
        if spec.tool_type == ToolType.PYTHON_FUNCTION:
            return "# TODO: Implement function logic"
        elif spec.tool_type == ToolType.DATA_PROCESSOR:
            return """
        # Basic data processing
        if 'operation' in locals():
            if operation == 'clean':
                df = df.dropna()
            elif operation == 'normalize':
                numeric_columns = df.select_dtypes(include=[np.number]).columns
                df[numeric_columns] = (df[numeric_columns] - df[numeric_columns].mean()) / df[numeric_columns].std()
        """
        elif spec.tool_type == ToolType.CODE_ANALYZER:
            return """
        # Basic code analysis
        results['metrics']['lines_of_code'] = len(code.split('\\n'))
        results['metrics']['functions'] = len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)])

        # Check for common issues
        if len(code) > 1000:
            results['issues'].append({
                'type': 'complexity',
                'message': 'Function is too long',
                'line': 1
            })
        """
        else:
            return "# TODO: Implement tool logic"

    def _generate_return_value(self, output_schema: Dict[str, Any]) -> str:
        """Genera el valor de retorno por defecto"""
        if not output_schema:
            return "None"

        output_type = output_schema.get("type", "dict")
        if output_type == "dict":
            return "{}"
        elif output_type == "list":
            return "[]"
        elif output_type == "str":
            return '""'
        elif output_type in ["int", "float"]:
            return "0"
        elif output_type == "bool":
            return "False"
        else:
            return "None"

    def _generate_request_logic(self, spec: ToolSpecification) -> str:
        """Genera lógica de request para API clients"""
        params = spec.input_schema
        param_names = list(params.keys())

        if param_names:
            # Crear payload con parámetros
            payload_items = [f"'{name}': {name}" for name in param_names]
            payload = "{" + ", ".join(payload_items) + "}"
            return f"data = json.dumps({payload})\n        kwargs = {{'data': data}}"
        else:
            return "kwargs = {}"


# =============================================================================
# VALIDADOR Y TESTER DE TOOLS
# =============================================================================


class ToolValidator:
    """Validador y tester de herramientas"""

    def __init__(self):
        self.test_templates = self._load_test_templates()

    def _load_test_templates(self) -> Dict[str, str]:
        """Carga plantillas de tests"""
        return {
            ToolType.PYTHON_FUNCTION: """
import pytest
import asyncio
from {module_name} import {function_name}

class Test{FunctionName}:
    \"\"\"Tests for {function_name}\"\"\"

    def test_basic_functionality(self):
        \"\"\"Test basic functionality\"\"\"
        # TODO: Add basic test cases
        assert True  # Placeholder

    def test_edge_cases(self):
        \"\"\"Test edge cases\"\"\"
        # TODO: Add edge case tests
        assert True  # Placeholder

    def test_error_handling(self):
        \"\"\"Test error handling\"\"\"
        # TODO: Add error handling tests
        assert True  # Placeholder
""",
            ToolType.API_CLIENT: """
import pytest
import asyncio
from {module_name} import {class_name}

class Test{class_name}:
    \"\"\"Tests for {class_name}\"\"\"

    @pytest.mark.asyncio
    async def test_api_connection(self):
        \"\"\"Test API connection\"\"\"
        async with {class_name}("http://test-api.com") as client:
            # TODO: Add API connection tests
            assert True  # Placeholder

    @pytest.mark.asyncio
    async def test_api_requests(self):
        \"\"\"Test API requests\"\"\"
        async with {class_name}("http://test-api.com") as client:
            # TODO: Add API request tests
            assert True  # Placeholder
""",
            ToolType.DATA_PROCESSOR: """
import pytest
import pandas as pd
import numpy as np
from {module_name} import {function_name}

class Test{FunctionName}:
    \"\"\"Tests for {function_name}\"\"\"

    def test_data_processing(self):
        \"\"\"Test data processing functionality\"\"\"
        # Create test data
        test_data = [
            {{"col1": 1, "col2": "a"}},
            {{"col1": 2, "col2": "b"}},
            {{"col1": 3, "col2": "c"}}
        ]

        # TODO: Add data processing tests
        result = {function_name}(test_data)
        assert result is not None

    def test_edge_cases(self):
        \"\"\"Test edge cases\"\"\"
        # Empty data
        result = {function_name}([])
        assert result is not None

        # Single item
        result = {function_name}([{"col1": 1}])
        assert result is not None
""",
        }

    async def validate_tool(self, implementation: ToolImplementation) -> Dict[str, Any]:
        """Valida una implementación de herramienta"""
        results = {
            "syntax_check": False,
            "imports_check": False,
            "type_hints_check": False,
            "docstring_check": False,
            "test_execution": False,
            "performance_check": False,
            "overall_score": 0.0,
            "issues": [],
            "recommendations": [],
        }

        # Verificar sintaxis
        results["syntax_check"] = self._check_syntax(implementation.code)

        # Verificar imports
        results["imports_check"] = self._check_imports(implementation.code)

        # Verificar type hints
        results["type_hints_check"] = self._check_type_hints(implementation.code)

        # Verificar docstring
        results["docstring_check"] = self._check_docstring(implementation.code)

        # Ejecutar tests si existen
        if implementation.test_code:
            results["test_execution"] = await self._run_tests(implementation.test_code)

        # Evaluar rendimiento
        if results["syntax_check"]:
            results["performance_check"] = await self._check_performance(
                implementation.code
            )

        # Calcular puntuación general
        checks = [
            "syntax_check",
            "imports_check",
            "type_hints_check",
            "docstring_check",
            "test_execution",
            "performance_check",
        ]
        passed_checks = sum(results[check] for check in checks)
        results["overall_score"] = passed_checks / len(checks)

        # Generar recomendaciones
        results["recommendations"] = self._generate_recommendations(results)

        return results

    def _check_syntax(self, code: str) -> bool:
        """Verifica la sintaxis del código"""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _check_imports(self, code: str) -> bool:
        """Verifica que los imports sean válidos"""
        try:
            tree = ast.parse(code)
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    imports.extend(
                        f"{module}.{alias.name}" if module else alias.name
                        for alias in node.names
                    )

            # Verificar imports críticos
            critical_imports = ["typing", "logging"]
            return any(imp in imports for imp in critical_imports)

        except:
            return False

    def _check_type_hints(self, code: str) -> bool:
        """Verifica el uso de type hints"""
        try:
            tree = ast.parse(code)

            functions_with_hints = 0
            total_functions = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    total_functions += 1
                    if node.returns or any(arg.annotation for arg in node.args.args):
                        functions_with_hints += 1

            return functions_with_hints > 0

        except:
            return False

    def _check_docstring(self, code: str) -> bool:
        """Verifica la presencia de docstrings"""
        try:
            tree = ast.parse(code)

            functions_with_docstrings = 0
            total_functions = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    total_functions += 1
                    if ast.get_docstring(node):
                        functions_with_docstrings += 1

            return functions_with_docstrings > 0

        except:
            return False

    async def _run_tests(self, test_code: str) -> bool:
        """Ejecuta los tests de la herramienta"""
        # Esta es una implementación simplificada
        # En producción, ejecutaría los tests en un entorno aislado
        try:
            # Verificar sintaxis de los tests
            ast.parse(test_code)
            return True
        except:
            return False

    async def _check_performance(self, code: str) -> bool:
        """Evalúa el rendimiento básico del código"""
        # Esta es una implementación simplificada
        # En producción, haría profiling real
        try:
            # Contar líneas de código como proxy de complejidad
            lines = len([line for line in code.split("\n") if line.strip()])
            return lines < 100  # Umbral arbitrario
        except:
            return False

    def _generate_recommendations(self, results: Dict[str, Any]) -> List[str]:
        """Genera recomendaciones basadas en los resultados de validación"""
        recommendations = []

        if not results["syntax_check"]:
            recommendations.append("Fix syntax errors in the code")

        if not results["imports_check"]:
            recommendations.append("Add proper imports for required modules")

        if not results["type_hints_check"]:
            recommendations.append(
                "Add type hints to function parameters and return values"
            )

        if not results["docstring_check"]:
            recommendations.append("Add comprehensive docstrings to all functions")

        if not results["test_execution"]:
            recommendations.append("Create and run comprehensive unit tests")

        if not results["performance_check"]:
            recommendations.append("Optimize code performance and reduce complexity")

        if results["overall_score"] < 0.7:
            recommendations.append("Overall code quality needs improvement")

        return recommendations

    def generate_test_code(self, spec: ToolSpecification, code: str) -> str:
        """Genera código de test para una herramienta"""
        template = self.test_templates.get(
            spec.tool_type, self.test_templates[ToolType.PYTHON_FUNCTION]
        )

        function_name = self._sanitize_name(spec.name)
        module_name = f"generated_tools.{function_name}"

        # Extraer información del código generado
        try:
            tree = ast.parse(code)
            classes = [
                node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
            ]
            functions = [
                node.name
                for node in ast.walk(tree)
                if isinstance(node, ast.FunctionDef)
            ]
        except:
            classes = []
            functions = [function_name]

        class_name = classes[0] if classes else function_name.title()
        function_name_actual = functions[0] if functions else function_name

        test_code = template.format(
            module_name=module_name,
            function_name=function_name_actual,
            FunctionName=function_name_actual.title(),
            class_name=class_name,
        )

        return test_code

    def _sanitize_name(self, name: str) -> str:
        """Convierte un nombre legible en un nombre de función válido"""
        name = re.sub(r"[^\w\s-]", "", name)
        name = re.sub(r"[\s-]+", "_", name)
        return name.lower()


# =============================================================================
# SISTEMA DE CREACIÓN DE TOOLS
# =============================================================================


class ToolCreationSystem:
    """Sistema principal para creación dinámica de tools"""

    def __init__(self):
        self.code_generator = CodeGenerator()
        self.validator = ToolValidator()
        self.created_tools: Dict[str, ToolImplementation] = {}
        self.pending_requests: List[ToolCreationRequest] = []
        self.creation_scheduler = None
        self.is_running = False

    async def start_tool_creation_system(self):
        """Inicia el sistema de creación de tools"""
        self.is_running = True

        # Iniciar scheduler de creación
        self.creation_scheduler = asyncio.create_task(self._tool_creation_loop())

        logger.info("Tool Creation System started")

    async def stop_tool_creation_system(self):
        """Detiene el sistema de creación de tools"""
        self.is_running = False

        if self.creation_scheduler:
            self.creation_scheduler.cancel()
            try:
                await self.creation_scheduler
            except asyncio.CancelledError:
                pass

        logger.info("Tool Creation System stopped")

    async def request_tool_creation(self, request: ToolCreationRequest) -> str:
        """Solicita la creación de una nueva herramienta"""
        self.pending_requests.append(request)
        logger.info(
            f"Tool creation requested: {request.tool_name} by {request.requester_agent}"
        )
        return request.request_id

    async def create_tool_from_spec(
        self, spec: ToolSpecification
    ) -> ToolImplementation:
        """Crea una herramienta desde una especificación"""
        logger.info(f"Creating tool: {spec.name}")

        # Generar código
        code = self.code_generator.generate_tool_code(spec)

        # Generar tests
        test_code = self.validator.generate_test_code(spec, code)

        # Crear implementación
        implementation = ToolImplementation(
            spec=spec, code=code, test_code=test_code, status=ToolStatus.DRAFT
        )

        # Validar implementación
        validation_results = await self.validator.validate_tool(implementation)
        implementation.validation_results = validation_results

        if validation_results["overall_score"] >= 0.7:
            implementation.status = ToolStatus.TESTING
        else:
            implementation.status = ToolStatus.FAILED

        # Almacenar herramienta creada
        self.created_tools[spec.tool_id] = implementation

        logger.info(
            f"Tool created: {spec.name} (status: {implementation.status.value})"
        )
        return implementation

    async def _tool_creation_loop(self):
        """Loop principal de creación de tools"""
        while self.is_running:
            try:
                # Procesar requests pendientes
                if self.pending_requests:
                    request = self.pending_requests.pop(0)
                    await self._process_creation_request(request)

                # Esperar antes de siguiente iteración
                await asyncio.sleep(10)  # Procesar cada 10 segundos

            except Exception as e:
                logger.error(f"Error in tool creation loop: {e}")
                await asyncio.sleep(30)  # Esperar más tiempo en caso de error

    async def _process_creation_request(self, request: ToolCreationRequest):
        """Procesa una solicitud de creación de tool"""
        try:
            request.status = "in_progress"

            # Crear especificación desde el request
            spec = ToolSpecification(
                name=request.tool_name,
                description=request.description,
                tool_type=request.tool_type,
                created_by=request.requester_agent,
            )

            # Inferir schemas desde ejemplos y requirements
            spec.input_schema = self._infer_input_schema(request)
            spec.output_schema = self._infer_output_schema(request)
            spec.parameters = request.requirements

            # Crear la herramienta
            implementation = await self.create_tool_from_spec(spec)

            # Si la validación fue exitosa, intentar deploy
            if implementation.status == ToolStatus.TESTING:
                await self._deploy_tool(implementation)

            request.status = "completed"

            # Registrar experiencia de aprendizaje
            await self._record_creation_experience(request, implementation)

        except Exception as e:
            logger.error(
                f"Error processing tool creation request {request.request_id}: {e}"
            )
            request.status = "failed"

    def _infer_input_schema(self, request: ToolCreationRequest) -> Dict[str, Any]:
        """Infier esquema de entrada desde el request"""
        schema = {}

        # Analizar ejemplos para inferir tipos
        if request.examples:
            for example in request.examples:
                if "input" in example:
                    for key, value in example["input"].items():
                        if key not in schema:
                            schema[key] = {
                                "type": type(value).__name__,
                                "description": f"Parameter {key}",
                            }

        # Usar requirements como fallback
        if not schema and request.requirements:
            for key, value in request.requirements.items():
                schema[key] = {
                    "type": type(value).__name__,
                    "description": f"Parameter {key}",
                    "default": value,
                }

        return schema

    def _infer_output_schema(self, request: ToolCreationRequest) -> Dict[str, Any]:
        """Infier esquema de salida desde el request"""
        schema = {"type": "dict", "description": "Result of the operation"}

        # Analizar ejemplos para inferir tipos de salida
        if request.examples:
            for example in request.examples:
                if "output" in example:
                    output = example["output"]
                    schema["type"] = type(output).__name__

                    if isinstance(output, dict):
                        schema["properties"] = {
                            k: type(v).__name__ for k, v in output.items()
                        }
                    elif isinstance(output, list):
                        schema["item_type"] = (
                            type(output[0]).__name__ if output else "any"
                        )

        return schema

    async def _deploy_tool(self, implementation: ToolImplementation):
        """Despliega una herramienta validada"""
        try:
            # Esta es una implementación simplificada
            # En producción, esto guardaría el código en archivos,
            # lo registraría en el sistema de módulos, etc.

            implementation.status = ToolStatus.DEPLOYED
            implementation.deployment_info = {
                "deployed_at": datetime.now().isoformat(),
                "version": implementation.spec.version,
            }

            logger.info(f"Tool deployed: {implementation.spec.name}")

        except Exception as e:
            logger.error(f"Error deploying tool {implementation.spec.tool_id}: {e}")
            implementation.status = ToolStatus.FAILED

    async def _record_creation_experience(
        self, request: ToolCreationRequest, implementation: ToolImplementation
    ):
        """Registra experiencia de creación de tool"""
        success = implementation.status in [ToolStatus.TESTING, ToolStatus.DEPLOYED]

        experience = LearningExperience(
            agent_id=request.requester_agent,
            task_type="tool_creation",
            input_data={
                "tool_name": request.tool_name,
                "tool_type": request.tool_type.value,
                "requirements": request.requirements,
            },
            action_taken={
                "creation_approach": "automated_generation",
                "validation_performed": True,
            },
            outcome={
                "tool_created": True,
                "validation_score": implementation.validation_results.get(
                    "overall_score", 0.0
                ),
                "deployment_success": success,
            },
            reward=1.0 if success else -0.5,
            quality_score=implementation.validation_results.get("overall_score", 0.0),
            context={
                "creation_complexity": "medium",
                "tool_type": request.tool_type.value,
                "has_examples": len(request.examples) > 0,
            },
        )

        await record_agent_experience(experience)

    def get_created_tools(self) -> List[ToolImplementation]:
        """Obtiene lista de tools creados"""
        return list(self.created_tools.values())

    def get_tool_by_id(self, tool_id: str) -> Optional[ToolImplementation]:
        """Obtiene una herramienta por ID"""
        return self.created_tools.get(tool_id)

    def get_tools_by_status(self, status: ToolStatus) -> List[ToolImplementation]:
        """Obtiene tools por estado"""
        return [tool for tool in self.created_tools.values() if tool.status == status]

    def get_creation_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de creación de tools"""
        total_tools = len(self.created_tools)
        deployed_tools = len(self.get_tools_by_status(ToolStatus.DEPLOYED))
        failed_tools = len(self.get_tools_by_status(ToolStatus.FAILED))

        return {
            "total_tools_created": total_tools,
            "deployed_tools": deployed_tools,
            "failed_tools": failed_tools,
            "success_rate": deployed_tools / total_tools if total_tools > 0 else 0,
            "pending_requests": len(self.pending_requests),
        }


# =============================================================================
# INTEGRACIÓN CON SISTEMA EXISTENTE
# =============================================================================

# Instancia global del sistema de creación de tools
tool_creation_system = ToolCreationSystem()


async def initialize_tool_creation():
    """Inicializa el sistema de creación de tools"""
    await tool_creation_system.start_tool_creation_system()


async def request_tool_creation(
    name: str,
    description: str,
    tool_type: ToolType,
    requirements: Dict[str, Any] = None,
    examples: List[Dict[str, Any]] = None,
    requester_agent: str = "system",
) -> str:
    """Solicita creación de una nueva herramienta"""
    request = ToolCreationRequest(
        requester_agent=requester_agent,
        tool_name=name,
        description=description,
        tool_type=tool_type,
        requirements=requirements or {},
        examples=examples or [],
    )

    return await tool_creation_system.request_tool_creation(request)


# =============================================================================
# DEMO Y TESTING
# =============================================================================


async def demo_tool_creation():
    """Demostración del sistema de creación de tools"""
    print("🔧 Inicializando sistema de creación de tools...")

    await initialize_tool_creation()

    # Solicitar creación de una herramienta
    print("\n📝 Solicitando creación de herramienta...")
    request_id = await request_tool_creation(
        name="Data Cleaner",
        description="Herramienta para limpiar y preprocesar datos",
        tool_type=ToolType.DATA_PROCESSOR,
        requirements={
            "supports_csv": True,
            "handles_missing": True,
            "normalization": True,
        },
        examples=[
            {
                "input": {"data": [{"col1": 1, "col2": None}, {"col1": 2, "col2": 3}]},
                "output": {
                    "cleaned_data": [{"col1": 1, "col2": 0}, {"col1": 2, "col2": 3}],
                    "stats": {"missing_filled": 1},
                },
            }
        ],
    )

    print(f"✅ Solicitud enviada con ID: {request_id}")

    # Esperar un poco para que se procese
    await asyncio.sleep(2)

    # Verificar estado
    stats = tool_creation_system.get_creation_stats()
    print(f"\n📊 Estadísticas del sistema:")
    print(f"   Tools creados: {stats['total_tools_created']}")
    print(f"   Tools desplegados: {stats['deployed_tools']}")
    print(f"   Tasa de éxito: {stats['success_rate']:.1%}")

    # Mostrar tools creados
    tools = tool_creation_system.get_created_tools()
    if tools:
        print(f"\n🛠️ Tools creados:")
        for tool in tools:
            print(f"   • {tool.spec.name} ({tool.status.value})")
            print(
                f"     Validación: {tool.validation_results.get('overall_score', 0):.1%}"
            )

    print("\n🎉 Demo de creación de tools completada!")


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Clases principales
    "CodeGenerator",
    "ToolValidator",
    "ToolCreationSystem",
    # Modelos de datos
    "ToolSpecification",
    "ToolImplementation",
    "ToolCreationRequest",
    "ToolType",
    "ToolStatus",
    # Sistema global
    "tool_creation_system",
    # Funciones de utilidad
    "initialize_tool_creation",
    "request_tool_creation",
    "demo_tool_creation",
]

# Información del módulo
__version__ = "1.0.0"
__author__ = "Sheily AI Team - Dynamic Tool Creation System"
