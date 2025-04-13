from manimlib import *
import numpy as np

class rc(Scene):
    def sign_not_zero(self, v):
        return 1.0 if v >= 0.0 else -1.0

    def oct_decode(self, f):
        f = f * 2.0 - 1.0
        n = np.array([f[0], f[1], 1.0 - np.abs(f[0]) - np.abs(f[1])])
        if n[2] < 0:
            n[:2] = (1.0 - np.abs(np.array([n[1], n[0]]))) * np.array([self.sign_not_zero(n[0]), self.sign_not_zero(n[1])])
        return n / np.linalg.norm(n)

    def generate_uniform_directions(self, resolution=4):
        directions = []
        for i in range(resolution):
            for j in range(resolution):
                u = (i + 0.5) / resolution
                v = (j + 0.5) / resolution
                dir = self.oct_decode(np.array([u, v]))
                directions.append(dir)
        return directions

    def create_wireframe_sphere_with_rays(self, center, radius, sphere_color=RED, ray_color=YELLOW, ray_resolution=4):
        slices = 6
        stacks = 4
        sphere_group = VGroup()
        wireframe = VGroup()

        for i in range(stacks):
            theta1 = (i / stacks) * PI
            theta2 = ((i + 1) / stacks) * PI
            for j in range(slices):
                phi1 = (j / slices) * 2 * PI
                phi2 = ((j + 1) / slices) * 2 * PI

                p1 = center + radius * np.array([np.sin(theta1) * np.cos(phi1), np.cos(theta1), np.sin(theta1) * np.sin(phi1)])
                p2 = center + radius * np.array([np.sin(theta2) * np.cos(phi1), np.cos(theta2), np.sin(theta2) * np.sin(phi1)])
                p3 = center + radius * np.array([np.sin(theta2) * np.cos(phi2), np.cos(theta2), np.sin(theta2) * np.sin(phi2)])
                p4 = center + radius * np.array([np.sin(theta1) * np.cos(phi2), np.cos(theta1), np.sin(theta1) * np.sin(phi2)])

                wireframe.add(
                    Line(p1, p2, color=sphere_color, stroke_width=0.7),
                    Line(p2, p3, color=sphere_color, stroke_width=0.7),
                    Line(p3, p4, color=sphere_color, stroke_width=0.7),
                    Line(p4, p1, color=sphere_color, stroke_width=0.7),
                )

        directions = self.generate_uniform_directions(ray_resolution)
        rays = VGroup()
        ray_length = radius * 2.0
        for direction in directions:
            start_point = center
            end_point = center + ray_length * direction
            ray = Arrow(start_point, end_point, buff=0, color=ray_color, stroke_width=0.4)
            rays.add(ray)

        sphere_group.add(wireframe, rays)
        return sphere_group

    def draw_2x2_grid(self, grid_width, grid_height):
        rows, cols = 2, 2
        spacing_x = grid_width / cols
        spacing_y = grid_height / rows
        radius = min(spacing_x, spacing_y) / 5
        z = 16
        spheres = VGroup()
        for row in range(rows):
            for col in range(cols):
                x = -grid_width / 2 + spacing_x / 2 + col * spacing_x
                y = grid_height / 2 - spacing_y / 2 - row * spacing_y
                center = np.array([x, y, z])
                sphere = self.create_wireframe_sphere_with_rays(center, radius, PURPLE, BLUE, ray_resolution=16)
                spheres.add(sphere)
        return spheres

    def draw_4x4_grid(self, grid_width, grid_height):
        rows, cols = 4, 4
        spacing_x = grid_width / cols
        spacing_y = grid_height / rows
        radius = min(spacing_x, spacing_y) / 6
        z = 5
        spheres = VGroup()
        for row in range(rows):
            for col in range(cols):
                x = -grid_width / 2 + spacing_x / 2 + col * spacing_x
                y = grid_height / 2 - spacing_y / 2 - row * spacing_y
                center = np.array([x, y, z])
                sphere = self.create_wireframe_sphere_with_rays(center, radius, BLUE, GREEN, ray_resolution=8)
                spheres.add(sphere)
        return spheres

    def draw_8x8_grid(self, grid_width, grid_height):
        rows, cols = 8, 8
        spacing_x = grid_width / cols
        spacing_y = grid_height / rows
        radius = min(spacing_x, spacing_y) / 10
        z = 0
        spheres = VGroup()
        for row in range(rows):
            for col in range(cols):
                x = -grid_width / 2 + spacing_x / 2 + col * spacing_x
                y = grid_height / 2 - spacing_y / 2 - row * spacing_y
                center = np.array([x, y, z])
                sphere = self.create_wireframe_sphere_with_rays(center, radius, RED, ORANGE, ray_resolution=4)
                spheres.add(sphere)
        return spheres

    def construct(self):
        self.camera.frame.set_euler_angles(
            theta=-90 * DEGREES,
            phi=0 * DEGREES
        )
        self.camera.frame.set_height(80)
        self.camera.frame.move_to(ORIGIN + [0, 0, 3])

        grid_width = 64
        grid_height = 64

        # Draw grids
        large_grid = self.draw_2x2_grid(grid_width, grid_height)
        medium_grid = self.draw_4x4_grid(grid_width, grid_height)
        small_grid = self.draw_8x8_grid(grid_width, grid_height)

        self.add(large_grid)
        self.wait(0.5)
        self.add(medium_grid)
        self.wait(0.5)
        self.add(small_grid)
        self.wait(0.5)

        # all_spheres = VGroup(large_grid, medium_grid, small_grid)

        target_direction = np.array([0, 0, 1])

        def get_probe_center(sphere_group):
            wireframe = sphere_group[0]
            if len(wireframe) == 0:
                return np.array([0, 0, 0])
            return wireframe[0].get_start()

        # Select exactly the center probe from the 8x8 grid
        # In an 8x8 grid, index 27 corresponds to row 3, column 3
        # which is one of the center probes
        selected_small_probe = small_grid[27]
        
        # Find best ray in the selected probe
        best_ray = None
        best_dot = -1
        
        for ray in selected_small_probe[1]:
            direction = ray.get_end() - ray.get_start()
            direction /= np.linalg.norm(direction)
            dot = np.dot(direction, target_direction)
            if dot > best_dot:
                best_dot = dot
                best_ray = ray

        selected_ray_vector = best_ray.get_end() - best_ray.get_start()
        selected_ray_vector /= np.linalg.norm(selected_ray_vector)

        # Highlight best 8x8 ray, remove others in probe
        self.play(best_ray.animate.set_color(RED).set_stroke(width=2))
        self.wait(1)
        other_rays = [ray for ray in selected_small_probe[1] if ray != best_ray]
        self.remove(*other_rays)
        self.wait(1) 

        def find_closest_probes(source_center, candidate_grid, count):
            centers = [get_probe_center(p) for p in candidate_grid]
            dists = [np.linalg.norm(center - source_center) for center in centers]
            closest_indices = np.argsort(dists)[:count]
            return [candidate_grid[i] for i in closest_indices]

        def find_most_similar_rays(probe, direction, count):
            rays = probe[1]
            similarities = []
            for ray in rays:
                vec = ray.get_end() - ray.get_start()
                vec /= np.linalg.norm(vec)
                dot = np.dot(vec, direction)
                similarities.append((ray, dot))
            return [ray for ray, _ in sorted(similarities, key=lambda x: -x[1])[:count]]

        # Find the 4 closest medium probes (instead of just 1)
        small_center = get_probe_center(selected_small_probe)
        closest_medium = find_closest_probes(small_center, medium_grid, 4)
        medium_selected_rays = []

        for probe in closest_medium:
            similar_rays = find_most_similar_rays(probe, selected_ray_vector, 4)
            medium_selected_rays.extend((ray, ray.get_end() - ray.get_start()) for ray in similar_rays)

            # Remove other rays
            all_rays = set(probe[1])
            rays_to_keep = set(similar_rays)
            self.remove(*(all_rays - rays_to_keep))

        self.play(*[ray.animate.set_color(RED).set_stroke(width=2) for ray, _ in medium_selected_rays])
        self.wait(1)

        large_selected_rays = []
        for ray, dir_vec in medium_selected_rays:
            dir_vec /= np.linalg.norm(dir_vec)
            center = ray.get_start()
            closest_large = find_closest_probes(center, large_grid, 4)
            for probe in closest_large:
                similar_rays = find_most_similar_rays(probe, dir_vec, 4)
                large_selected_rays.extend(similar_rays)

                # Remove other rays
                all_rays = set(probe[1])
                rays_to_keep = set(similar_rays)
                self.remove(*(all_rays - rays_to_keep))

        self.play(*[ray.animate.set_color(RED).set_stroke(width=2) for ray in large_selected_rays])
        self.wait(1)