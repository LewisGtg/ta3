import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os

class PerspectiveTransformer:
    def __init__(self, root):
        self.root = root
        self.root.title("Transformador de Perspectiva")
        self.root.geometry("1200x700")
        
        # Variáveis para armazenar imagens e pontos
        self.original_image = None
        self.displayed_image = None
        self.working_image = None
        self.points = []
        self.destination_points = None
        self.is_drawing = False
        self.current_mode = "document"  # Modos: document, aerial, painting
        
        # Criar a interface
        self.create_widgets()
    
    def create_widgets(self):
        # Frame principal
        main_frame = tk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Frame de controle à esquerda
        control_frame = tk.Frame(main_frame, width=200)
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Botões de controle
        tk.Button(control_frame, text="Abrir Imagem", command=self.open_image, width=20).pack(pady=5)
        
        # Seleção de modo
        mode_frame = tk.LabelFrame(control_frame, text="Modo de Transformação")
        mode_frame.pack(pady=10, fill=tk.X)
        
        self.mode_var = tk.StringVar(value="document")
        modes = [
            ("Documento", "document"),
            ("Vista Aérea", "aerial"),
            ("Pintura", "painting"),
            ("Personalizado", "custom")
        ]
        
        for text, mode in modes:
            tk.Radiobutton(mode_frame, text=text, variable=self.mode_var, 
                          value=mode, command=self.change_mode).pack(anchor=tk.W, pady=2)
        
        # Botões de ação
        tk.Button(control_frame, text="Limpar Pontos", command=self.clear_points, width=20).pack(pady=5)
        tk.Button(control_frame, text="Transformar", command=self.transform_image, width=20).pack(pady=5)
        tk.Button(control_frame, text="Salvar Imagem", command=self.save_image, width=20).pack(pady=5)
        
        # Instruções
        instruction_frame = tk.LabelFrame(control_frame, text="Instruções")
        instruction_frame.pack(pady=10, fill=tk.X)
        instructions = "1. Abra uma imagem\n2. Selecione o modo\n3. Clique em 4 pontos na imagem\n4. Clique em 'Transformar'\n5. Salve o resultado"
        tk.Label(instruction_frame, text=instructions, justify=tk.LEFT).pack(pady=5)
        
        # Canvas para mostrar a imagem
        self.canvas_frame = tk.Frame(main_frame)
        self.canvas_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="lightgray")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Vincular eventos do mouse
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<Configure>", self.on_resize)
    
    def open_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Imagens", "*.jpg *.jpeg *.png *.bmp *.tif")]
        )
        
        if file_path:
            self.original_image = cv2.imread(file_path)
            if self.original_image is None:
                messagebox.showerror("Erro", "Não foi possível abrir a imagem.")
                return
            
            self.working_image = self.original_image.copy()
            self.clear_points()
            self.display_image(self.working_image)
    
    def display_image(self, image):
        # Converter de BGR para RGB para exibição
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Redimensionar para caber no canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        if canvas_width > 1 and canvas_height > 1:
            h, w = image_rgb.shape[:2]
            scale = min(canvas_width / w, canvas_height / h)
            
            new_width = int(w * scale)
            new_height = int(h * scale)
            
            resized_image = cv2.resize(image_rgb, (new_width, new_height))
            
            # Converter para formato Tkinter
            self.displayed_image = ImageTk.PhotoImage(image=Image.fromarray(resized_image))
            
            # Centralizar no canvas
            x_position = (canvas_width - new_width) // 2
            y_position = (canvas_height - new_height) // 2
            
            self.canvas.delete("all")
            self.canvas.create_image(x_position, y_position, anchor=tk.NW, image=self.displayed_image)
            
            # Ajustar a escala para os pontos marcados
            self.display_scale = scale
            self.x_offset = x_position
            self.y_offset = y_position
            
            # Redesenhar os pontos
            self.redraw_points()
    
    def on_resize(self, event):
        if self.working_image is not None:
            self.display_image(self.working_image)
    
    def on_click(self, event):
        if self.original_image is None:
            messagebox.showinfo("Aviso", "Abra uma imagem primeiro.")
            return
        
        if len(self.points) >= 4 and self.mode_var.get() != "custom":
            messagebox.showinfo("Aviso", "Já foram selecionados 4 pontos. Clique em 'Limpar Pontos' para recomeçar.")
            return
        
        # Converter coordenadas do canvas para coordenadas da imagem
        x = (event.x - self.x_offset) / self.display_scale
        y = (event.y - self.y_offset) / self.display_scale
        
        # Verificar se o clique está dentro da imagem
        h, w = self.original_image.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            self.points.append((int(x), int(y)))
            
            # Desenhar o ponto no canvas
            canvas_x = event.x
            canvas_y = event.y
            self.canvas.create_oval(canvas_x-5, canvas_y-5, canvas_x+5, canvas_y+5, fill="red", outline="white", tags="point")
            
            # Se for o segundo ou mais ponto, desenhar a linha conectando-os
            if len(self.points) > 1:
                prev_x, prev_y = self.points[-2]
                prev_canvas_x = prev_x * self.display_scale + self.x_offset
                prev_canvas_y = prev_y * self.display_scale + self.y_offset
                self.canvas.create_line(prev_canvas_x, prev_canvas_y, canvas_x, canvas_y, fill="yellow", width=2, tags="line")
            
            # Se for o quarto ponto, fechar o polígono
            if len(self.points) == 4:
                first_x, first_y = self.points[0]
                first_canvas_x = first_x * self.display_scale + self.x_offset
                first_canvas_y = first_y * self.display_scale + self.y_offset
                self.canvas.create_line(canvas_x, canvas_y, first_canvas_x, first_canvas_y, fill="yellow", width=2, tags="line")
    
    def redraw_points(self):
        self.canvas.delete("point", "line")
        
        for i, point in enumerate(self.points):
            x, y = point
            canvas_x = x * self.display_scale + self.x_offset
            canvas_y = y * self.display_scale + self.y_offset
            self.canvas.create_oval(canvas_x-5, canvas_y-5, canvas_x+5, canvas_y+5, fill="red", outline="white", tags="point")
            
            # Desenhar linhas entre os pontos
            if i > 0:
                prev_x, prev_y = self.points[i-1]
                prev_canvas_x = prev_x * self.display_scale + self.x_offset
                prev_canvas_y = prev_y * self.display_scale + self.y_offset
                self.canvas.create_line(prev_canvas_x, prev_canvas_y, canvas_x, canvas_y, fill="yellow", width=2, tags="line")
        
        # Fechar o polígono se houver 4 pontos
        if len(self.points) == 4:
            first_x, first_y = self.points[0]
            last_x, last_y = self.points[3]
            first_canvas_x = first_x * self.display_scale + self.x_offset
            first_canvas_y = first_y * self.display_scale + self.y_offset
            last_canvas_x = last_x * self.display_scale + self.x_offset
            last_canvas_y = last_y * self.display_scale + self.y_offset
            self.canvas.create_line(last_canvas_x, last_canvas_y, first_canvas_x, first_canvas_y, fill="yellow", width=2, tags="line")
    
    def clear_points(self):
        self.points = []
        if hasattr(self, 'canvas'):
            self.canvas.delete("point", "line")
    
    def change_mode(self):
        self.current_mode = self.mode_var.get()
        self.clear_points()
        
        if self.original_image is not None:
            self.working_image = self.original_image.copy()
            self.display_image(self.working_image)
    
    def transform_image(self):
        if len(self.points) != 4:
            messagebox.showinfo("Aviso", "Selecione exatamente 4 pontos para transformar a imagem.")
            return
        
        # Obter os pontos de origem em uma ordem específica (superior-esquerdo, superior-direito, inferior-direito, inferior-esquerdo)
        src_points = np.array(self.points, dtype=np.float32)
        
        # Calcular os pontos de destino com base no modo selecionado
        h, w = self.original_image.shape[:2]
        
        if self.current_mode == "document":
            # Para documentos, cria um retângulo com proporções adequadas
            # Calcular a largura e altura do quadrilátero de origem
            width_top = np.sqrt(((src_points[1][0] - src_points[0][0]) ** 2) + ((src_points[1][1] - src_points[0][1]) ** 2))
            width_bottom = np.sqrt(((src_points[2][0] - src_points[3][0]) ** 2) + ((src_points[2][1] - src_points[3][1]) ** 2))
            width = max(int(width_top), int(width_bottom))
            
            height_left = np.sqrt(((src_points[3][0] - src_points[0][0]) ** 2) + ((src_points[3][1] - src_points[0][1]) ** 2))
            height_right = np.sqrt(((src_points[2][0] - src_points[1][0]) ** 2) + ((src_points[2][1] - src_points[1][1]) ** 2))
            height = max(int(height_left), int(height_right))
            
            dst_points = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
            
            output_size = (width, height)
            
        elif self.current_mode == "aerial":
            # Para vista aérea, cria uma perspectiva de cima
            dst_points = np.array([
                [w * 0.1, h * 0.1],
                [w * 0.9, h * 0.1],
                [w * 0.9, h * 0.9],
                [w * 0.1, h * 0.9]
            ], dtype=np.float32)
            
            output_size = (w, h)
            
        elif self.current_mode == "painting":
            # Para pinturas, ajusta para dar uma perspectiva diferente
            dst_points = np.array([
                [w * 0.2, h * 0.2],
                [w * 0.8, h * 0.2],
                [w * 0.8, h * 0.8],
                [w * 0.2, h * 0.8]
            ], dtype=np.float32)
            
            output_size = (w, h)
            
        else:  # Custom
            # Usa os pontos mais extremos como referência
            min_x = min(p[0] for p in self.points)
            max_x = max(p[0] for p in self.points)
            min_y = min(p[1] for p in self.points)
            max_y = max(p[1] for p in self.points)
            
            width = int(max_x - min_x)
            height = int(max_y - min_y)
            
            dst_points = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype=np.float32)
            
            output_size = (width, height)
        
        # Calcular a matriz de transformação de perspectiva
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Aplicar a transformação
        warped = cv2.warpPerspective(self.original_image, M, output_size)
        
        # Atualizar a imagem
        self.working_image = warped
        self.display_image(self.working_image)
    
    def save_image(self):
        if self.working_image is None:
            messagebox.showinfo("Aviso", "Nenhuma imagem transformada para salvar.")
            return
        
        file_path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("Todos os arquivos", "*.*")]
        )
        
        if file_path:
            cv2.imwrite(file_path, self.working_image)
            messagebox.showinfo("Sucesso", f"Imagem salva como {os.path.basename(file_path)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = PerspectiveTransformer(root)
    root.mainloop()