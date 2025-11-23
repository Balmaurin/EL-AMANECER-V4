import ast
import os
from pathlib import Path
import re

def get_smart_description(filepath, root_dir):
    """
    Intenta obtener una descripción inteligente del archivo usando múltiples estrategias.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            
        description = ""
        source = "Unknown"
        
        # Estrategia 1: Docstring del Módulo (AST)
        try:
            tree = ast.parse(content)
            docstring = ast.get_docstring(tree)
            if docstring:
                # Limpiar y tomar el primer párrafo significativo
                lines = [l.strip() for l in docstring.split('\n') if l.strip()]
                if lines:
                    description = " ".join(lines[:3]) # Primeras 3 líneas
                    source = "Docstring"
                    return description, source
        except:
            pass

        # Estrategia 2: Comentarios iniciales (Fallback manual)
        if not description:
            lines = content.split('\n')
            comments = []
            for line in lines[:15]: # Mirar primeras 15 líneas
                line = line.strip()
                if line.startswith('#'):
                    comment = line.lstrip('#').strip()
                    # Ignorar shebangs o encoding
                    if not comment.startswith('!') and 'coding' not in comment:
                        comments.append(comment)
                elif line.startswith('"""') or line.startswith("'''"):
                    continue # Ya manejado por AST usualmente, pero por si acaso
            
            if comments:
                description = " ".join(comments[:2])
                source = "Comments"
                return description, source

        # Estrategia 3: Análisis de Código (Clases y Funciones principales)
        if not description:
            try:
                classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) if not node.name.startswith('_')]
                
                parts = []
                if classes:
                    parts.append(f"Define clases: {', '.join(classes[:3])}")
                if functions:
                    parts.append(f"Funciones: {', '.join(functions[:5])}")
                
                if parts:
                    description = ". ".join(parts)
                    source = "Code Analysis"
                    return description, source
            except:
                pass

        # Estrategia 4: Inferencia por Path y Nombre
        if not description:
            rel_path = Path(filepath).relative_to(root_dir)
            parts = list(rel_path.parts)
            filename = rel_path.stem.replace('_', ' ').title()
            
            context = []
            if 'tests' in parts or 'testing' in parts:
                context.append("Test script")
            if 'tools' in parts:
                context.append("Herramienta de utilidad")
            if 'api' in parts:
                context.append("Endpoint o interfaz API")
            if 'models' in parts:
                context.append("Definición de modelos de datos")
                
            description = f"Módulo '{filename}'."
            if context:
                description += f" Posible propósito: {', '.join(context)}."
            
            source = "Path Inference"
            return description, source

    except Exception as e:
        return f"Error analizando: {str(e)}", "Error"
        
    return "Análisis no concluyente", "Empty"

def main():
    root_dir = Path(os.getcwd())
    inventory_path = root_dir / 'INVENTARIO_COMPLETO_SCRIPTS.txt'
    output_path = root_dir / 'DOCUMENTACION_TECNICA_DETALLADA.md'
    
    if not inventory_path.exists():
        print(f"Error: No se encuentra {inventory_path}")
        return

    print("Generando documentación técnica inteligente...")
    
    with open(inventory_path, 'r', encoding='utf-8') as f:
        files = [line.strip() for line in f if line.strip()]

    with open(output_path, 'w', encoding='utf-8') as out:
        out.write("# 📚 Documentación Técnica Detallada de Scripts (Mejorada)\n\n")
        out.write(f"Generado automáticamente con análisis profundo. Total de archivos: {len(files)}\n\n")
        out.write("| Archivo | Descripción / Análisis | Fuente |\n")
        out.write("|---------|------------------------|--------|\n")
        
        for file_path in files:
            full_path = Path(file_path)
            if not full_path.is_absolute():
                full_path = root_dir / file_path
            
            if not full_path.exists():
                continue
                
            try:
                rel_path = full_path.relative_to(root_dir)
            except ValueError:
                rel_path = full_path.name
                
            description, source = get_smart_description(full_path, root_dir)
            
            # Limpieza para Markdown
            description = description.replace('|', '-').replace('\n', ' ')[:400] # Limitar largo
            
            # Iconos de fuente
            icon = "📄" # Docstring
            if source == "Comments": icon = "💬"
            if source == "Code Analysis": icon = "🔍"
            if source == "Path Inference": icon = "📂"
            if source == "Error": icon = "⚠️"
            
            out.write(f"| `{rel_path}` | {description} | {icon} {source} |\n")

    print(f"Documentación mejorada generada en: {output_path}")

if __name__ == "__main__":
    main()
