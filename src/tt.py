#!/usr/bin/env python3
import os

def count_sloc(folder='.'):
    total = 0
    details = []
    
    for root, dirs, files in os.walk(folder):
        # Исключаем виртуальные окружения и служебные папки
        if any(ex in root for ex in ['venv', 'env', '.git', '__pycache__', 'node_modules']):
            continue
            
        for f in files:
            if f.endswith(('.py', '.sql')) and f != 'count_lines.py':
                path = os.path.join(root, f)
                try:
                    with open(path, 'r', encoding='utf-8', errors='ignore') as file:
                        lines = file.readlines()
                        # Считаем только непустые строки, не являющиеся комментариями
                        code = [l for l in lines if l.strip() and not l.strip().startswith('#')]
                        count = len(code)
                        if count > 0:
                            details.append((f, count))
                            total += count
                except Exception:
                    continue
    
    return total, details

if __name__ == '__main__':
    total, details = count_sloc()
    print(f"\n📊 Результат подсчёта (SLOC — чистый код):")
    print(f"{'Файл':<40} {'Строки':>6}")
    print("-" * 47)
    for name, cnt in sorted(details, key=lambda x: -x[1]):
        print(f"{name:<40} {cnt:>6}")
    print("-" * 47)
    print(f"{'ИТОГО':<40} {total:>6}")
    print(f"\n✅ В курсовую вписывайте: {total} строк")