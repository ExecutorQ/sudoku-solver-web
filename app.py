from flask import Flask, request, jsonify, render_template
import os
import cv2
import numpy as np
from PIL import Image
from paddleocr import PaddleOCR
from pysat.solvers import Glucose3
from werkzeug.utils import secure_filename
import base64
import io

app = Flask(__name__)

# ---------- 設定區 ----------
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
CROP_BOX = (28, 538, 1141, 1654)  # 固定裁切座標
ocr = PaddleOCR(use_angle_cls=False, lang='en')

# ---------- 數獨解題邏輯 ----------
def sudoku_to_cnf(grid):
    cnf = []
    def v(i, j, d): return 81 * (i-1) + 9 * (j-1) + d

    for i in range(1, 10):
        for j in range(1, 10):
            if grid[i-1][j-1] != 0:
                cnf.append([v(i, j, grid[i-1][j-1])])

    for i in range(1, 10):
        for j in range(1, 10):
            cnf.append([v(i, j, d) for d in range(1, 10)])
            for d in range(1, 10):
                for dp in range(d+1, 10):
                    cnf.append([-v(i, j, d), -v(i, j, dp)])

    for i in range(1, 10):
        for d in range(1, 10):
            for j1 in range(1, 10):
                for j2 in range(j1+1, 10):
                    cnf.append([-v(i, j1, d), -v(i, j2, d)])

    for j in range(1, 10):
        for d in range(1, 10):
            for i1 in range(1, 10):
                for i2 in range(i1+1, 10):
                    cnf.append([-v(i1, j, d), -v(i2, j, d)])

    for bi in range(0, 3):
        for bj in range(0, 3):
            for d in range(1, 10):
                cells = [(bi*3+i+1, bj*3+j+1) for i in range(3) for j in range(3)]
                for a in range(len(cells)):
                    for b in range(a+1, len(cells)):
                        i1, j1 = cells[a]
                        i2, j2 = cells[b]
                        cnf.append([-v(i1, j1, d), -v(i2, j2, d)])
    return cnf

def solve_sudoku(grid):
    cnf = sudoku_to_cnf(grid)
    solver = Glucose3()
    for clause in cnf:
        solver.add_clause(clause)
    solved = solver.solve()
    model = solver.get_model()
    if not solved or not model:
        return None
    solution = [[0]*9 for _ in range(9)]
    for val in model:
        if val > 0:
            d = (val - 1) % 9 + 1
            j = ((val - 1) // 9) % 9
            i = ((val - 1) // 81)
            solution[i][j] = d
    return solution

# ---------- 裁切與辨識 ----------
def fixed_crop_and_ocr(image_path):
    image = cv2.imread(image_path)
    x1, y1, x2, y2 = CROP_BOX
    cropped = image[y1:y2, x1:x2]
    h, w = cropped.shape[:2]
    cell_h = h // 9
    cell_w = w // 9

    grid = []
    cell_imgs = []

    for i in range(9):
        row = []
        for j in range(9):
            cell = cropped[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
            gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
            _, threshed = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            result = ocr.ocr(threshed, cls=False)
            if result and result[0]:
                text = result[0][0][1][0]
                row.append(int(text) if text.isdigit() else 0)
            else:
                row.append(0)

            _, buf = cv2.imencode('.png', cell)
            img_b64 = base64.b64encode(buf).decode('utf-8')
            cell_imgs.append(f"data:image/png;base64,{img_b64}")
        grid.append(row)

    return grid, cell_imgs

# ---------- 路由 ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/recognize", methods=["POST"])
def recognize():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    file = request.files["image"]
    filename = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)
    grid, cell_imgs = fixed_crop_and_ocr(path)
    digits = [val for row in grid for val in row]
    return jsonify({"digits": digits, "cells": cell_imgs})

@app.route("/solve", methods=["POST"])
def solve():
    data = request.get_json()
    digits = data.get("digits", [])
    if len(digits) != 81:
        return jsonify({"error": "Invalid input"}), 400
    grid = [digits[i*9:(i+1)*9] for i in range(9)]
    solution = solve_sudoku(grid)
    if solution is None:
        return jsonify({"error": "Unsolvable"}), 400
    flat = [n for row in solution for n in row]
    return jsonify({"solution": flat})

if __name__ == "__main__":
    app.run(debug=True)
