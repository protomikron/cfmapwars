#!/usr/bin/env python3
import sys
import os
import random
import math
import ctypes
from collections import defaultdict

import sdl2
import sdl2.ext
import sdl2.sdlttf
import numpy as np
from scipy.spatial import Voronoi, cKDTree
from osgeo import gdal, ogr

gdal.UseExceptions()

# Game constants
WINDOW_WIDTH = 1700
WINDOW_HEIGHT = 1100
MAX_DICE = 8
MAX_PLAYERS = 4
BASE_REINFORCEMENTS = 3

# Colors (RGBA)
PLAYER_COLORS = [
    (220, 20, 60, 255),    # Crimson Red
    (30, 144, 255, 255),   # Dodger Blue
    (50, 205, 50, 255),    # Lime Green
    (255, 215, 0, 255)     # Gold
]

NEUTRAL_COLOR = (180, 180, 180, 255)
BORDER_COLOR = (0, 0, 0, 255)
HIGHLIGHT_COLOR = (255, 255, 255, 255)
BG_COLOR = (0, 0, 0, 255)
TEXT_COLOR = (20, 20, 20, 255)

# Cache of pre-rendered dice patterns
DICE_PATTERNS = {}

class SpatialIndex:
    """Efficient spatial index for territory lookup"""
    def __init__(self, territories):
        self.territories = territories
        self.points = np.array([t.center for t in territories.values()])
        self.ids = list(territories.keys())
        self.kdtree = cKDTree(self.points)
        
    def find_nearest(self, point, k=1):
        """Find k nearest territories to point"""
        distances, indices = self.kdtree.query(point, k=k)
        return [self.ids[i] for i in indices]

# Pre-compute common polygon paths and patterns
class RenderCache:
    """Cache for rendering optimizations"""
    def __init__(self, renderer):
        self.renderer = renderer
        self.polygon_textures = {}  # Cache polygon textures by shape hash
        self.dice_textures = {}     # Cache dice textures
        
    def get_polygon_texture(self, points, color):
        """Get or create a texture for a polygon"""
        # Create a hash of the polygon points and color
        points_tuple = tuple(map(tuple, points))
        key = (points_tuple, color)
        
        if key in self.polygon_textures:
            return self.polygon_textures[key]
            
        # Compute bounding box for the polygon
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        width = int(max_x - min_x) + 1
        height = int(max_y - min_y) + 1
        
        # Create a temporary target for rendering
        target = sdl2.SDL_CreateTexture(
            self.renderer,
            sdl2.SDL_PIXELFORMAT_RGBA8888,
            sdl2.SDL_TEXTUREACCESS_TARGET,
            width, height
        )
        
        # Save the current render target
        old_target = ctypes.c_void_p()
        sdl2.SDL_GetRenderTarget(self.renderer, ctypes.byref(old_target))
        
        # Set the new render target
        sdl2.SDL_SetRenderTarget(self.renderer, target)
        
        # Clear with transparent color
        sdl2.SDL_SetRenderDrawColor(self.renderer, 0, 0, 0, 0)
        sdl2.SDL_RenderClear(self.renderer)
        
        # Offset points to fit in the texture
        adjusted_points = [(x - min_x, y - min_y) for x, y in points]
        
        # Draw the polygon on the texture
        draw_filled_polygon(self.renderer, adjusted_points, color)
        
        # Set the texture blend mode to allow transparency
        sdl2.SDL_SetTextureBlendMode(target, sdl2.SDL_BLENDMODE_BLEND)
        
        # Restore the old render target
        sdl2.SDL_SetRenderTarget(self.renderer, old_target.value)
        
        # Store in cache
        self.polygon_textures[key] = (target, min_x, min_y, width, height)
        
        return self.polygon_textures[key]
    
    def get_dice_texture(self, dice_count, color):
        """Get or create a texture for dice"""
        key = (dice_count, color)
        
        if key in self.dice_textures:
            return self.dice_textures[key]
            
        # Size for the dice texture
        size = 32
        
        # Create a temporary target
        target = sdl2.SDL_CreateTexture(
            self.renderer,
            sdl2.SDL_PIXELFORMAT_RGBA8888,
            sdl2.SDL_TEXTUREACCESS_TARGET,
            size, size
        )
        
        # Save the current render target
        old_target = ctypes.c_void_p()
        sdl2.SDL_GetRenderTarget(self.renderer, ctypes.byref(old_target))
        
        # Set the new target
        sdl2.SDL_SetRenderTarget(self.renderer, target)
        
        # Clear with transparent color
        sdl2.SDL_SetRenderDrawColor(self.renderer, 0, 0, 0, 0)
        sdl2.SDL_RenderClear(self.renderer)
        
        # Draw the dice
        center_x, center_y = size // 2, size // 2
        bright_color = tuple(min(c + 50, 255) for c in color[:3]) + (color[3],)
        
        # Draw dice background
        #draw_filled_circle(self.renderer, center_x, center_y, 12, bright_color)
        #draw_circle(self.renderer, center_x, center_y, 12, BORDER_COLOR)
        
        # Draw pips
        if dice_count <= 6:
            pip_radius = 3
            pip_offset = 6
            
            pip_positions = []
            if dice_count == 1:
                pip_positions = [(center_x, center_y)]
            elif dice_count == 2:
                pip_positions = [(center_x - pip_offset, center_y - pip_offset), 
                                (center_x + pip_offset, center_y + pip_offset)]
            elif dice_count == 3:
                pip_positions = [(center_x - pip_offset, center_y - pip_offset), 
                                (center_x, center_y), 
                                (center_x + pip_offset, center_y + pip_offset)]
            elif dice_count == 4:
                pip_positions = [(center_x - pip_offset, center_y - pip_offset), 
                                (center_x + pip_offset, center_y - pip_offset),
                                (center_x - pip_offset, center_y + pip_offset), 
                                (center_x + pip_offset, center_y + pip_offset)]
            elif dice_count == 5:
                pip_positions = [(center_x - pip_offset, center_y - pip_offset), 
                                (center_x + pip_offset, center_y - pip_offset),
                                (center_x, center_y),
                                (center_x - pip_offset, center_y + pip_offset), 
                                (center_x + pip_offset, center_y + pip_offset)]
            elif dice_count == 6:
                pip_positions = [(center_x - pip_offset, center_y - pip_offset), 
                                (center_x + pip_offset, center_y - pip_offset),
                                (center_x - pip_offset, center_y), 
                                (center_x + pip_offset, center_y),
                                (center_x - pip_offset, center_y + pip_offset), 
                                (center_x + pip_offset, center_y + pip_offset)]
            
            for px, py in pip_positions:
                #draw_filled_circle(self.renderer, int(px), int(py), pip_radius, BORDER_COLOR)
                pass
        else:
            # For dice > 6, draw number as big dots in center
            #draw_filled_circle(self.renderer, center_x, center_y, 6, BORDER_COLOR)
            pass
        
        # Set blend mode
        sdl2.SDL_SetTextureBlendMode(target, sdl2.SDL_BLENDMODE_BLEND)
        
        # Restore old target
        sdl2.SDL_SetRenderTarget(self.renderer, old_target.value)
        
        # Store in cache
        self.dice_textures[key] = target
        
        return target
    
    def cleanup(self):
        """Free all cached textures"""
        for texture, _, _, _, _ in self.polygon_textures.values():
            sdl2.SDL_DestroyTexture(texture)
        
        for texture in self.dice_textures.values():
            sdl2.SDL_DestroyTexture(texture)
        
        self.polygon_textures.clear()
        self.dice_textures.clear()

# Optimized helper functions for polygon drawing
def draw_polygon(renderer, points, color):
    """Draw polygon outline"""
    sdl2.SDL_SetRenderDrawColor(renderer, color[0], color[1], color[2], color[3])
    
    # Convert points to integers and use SDL_RenderDrawLines for better performance
    # SDL_RenderDrawLines draws connected lines between all points provided
    # Create a C array of SDL_Point structures
    sdl_points = (sdl2.SDL_Point * (len(points) + 1))()
    
    for i, (x, y) in enumerate(points):
        sdl_points[i].x = int(x)
        sdl_points[i].y = int(y)
    
    # Connect back to the first point to close the polygon
    sdl_points[len(points)].x = int(points[0][0])
    sdl_points[len(points)].y = int(points[0][1])
    
    # Draw all lines at once
    sdl2.SDL_RenderDrawLines(renderer, sdl_points, len(points) + 1)

def draw_filled_polygon(renderer, points, color):
    """Draw filled polygon using an optimized scanline algorithm with edge tables"""
    if len(points) < 3:
        return
        
    # Convert to numpy arrays for faster processing
    points_array = np.array(points, dtype=np.float32)
    
    # Set draw color
    sdl2.SDL_SetRenderDrawColor(renderer, color[0], color[1], color[2], color[3])
    
    # Find the y-coordinate range
    y_min = int(np.min(points_array[:, 1]))
    y_max = int(np.max(points_array[:, 1]))
    
    # Build edge table - group edges by their minimum y-coord
    edge_table = defaultdict(list)
    
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        
        # Ensure y1 <= y2
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        
        # Skip horizontal edges
        if int(y1) == int(y2):
            continue
        
        # Calculate inverse slope (dx/dy)
        inverse_slope = (x2 - x1) / (y2 - y1) if y2 != y1 else 0
        
        # Add edge to edge table
        edge_table[int(y1)].append((x1, int(y2), inverse_slope))
    
    # Active edge list
    active_edges = []
    
    # Process each scanline from top to bottom
    for y in range(y_min, y_max + 1):
        # Add edges starting at this scanline to active edge list
        if y in edge_table:
            active_edges.extend(edge_table[y])
        
        # Remove edges that end at this scanline
        active_edges = [edge for edge in active_edges if edge[1] > y]
        
        # Sort active edges by x-coordinate
        active_edges.sort(key=lambda edge: edge[0])
        
        # Draw horizontal lines between pairs of x-coordinates
        for i in range(0, len(active_edges), 2):
            if i + 1 < len(active_edges):
                x1 = int(active_edges[i][0])
                x2 = int(active_edges[i + 1][0])
                sdl2.SDL_RenderDrawLine(renderer, x1, y, x2, y)
        
        # Update x-coordinates for the next scanline
        for i in range(len(active_edges)):
            x, y_max, inverse_slope = active_edges[i]
            active_edges[i] = (x + inverse_slope, y_max, inverse_slope)

def draw_circle(renderer, x, y, radius, color):
    """Draw circle outline - optimized version"""
    sdl2.SDL_SetRenderDrawColor(renderer, color[0], color[1], color[2], color[3])
    
    # Use midpoint circle algorithm but with batched rendering for better performance
    points = []
    
    # Calculate all points on the circle
    for angle in range(0, 360, 5):  # Every 5 degrees for a smooth circle
        rad = math.radians(angle)
        px = int(x + radius * math.cos(rad))
        py = int(y + radius * math.sin(rad))
        points.append((px, py))
    
    # Draw connected lines for the circle
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        sdl2.SDL_RenderDrawLine(renderer, x1, y1, x2, y2)

def draw_filled_circle(renderer, x, y, radius, color):
    """Draw filled circle - optimized version"""
    sdl2.SDL_SetRenderDrawColor(renderer, color[0], color[1], color[2], color[3])
    
    # Ensure radius is positive and all values are integers
    radius = max(1, int(radius))
    x, y = int(x), int(y)
    
    # Use horizontal scanlines for filling
    for dy in range(-radius, radius + 1):
        # Calculate width at this height using the circle equation
        dx = int(math.sqrt(radius * radius - dy * dy)) if (radius * radius - dy * dy) >= 0 else 0
        sdl2.SDL_RenderDrawLine(renderer, x - dx, y + dy, x + dx, y + dy)

class Territory:
    """Represents a single territory/region on the map"""
    
    def __init__(self, id, name, polygon, center):
        self.id = id                  # Unique identifier
        self.name = name              # Name based on NUTS-3 data
        self.polygon = polygon        # List of (x,y) points defining the boundary
        self.center = center          # Center point (x,y)
        self.owner = -1               # -1 for neutral, 0+ for player ID
        self.dice = random.randint(1, 3)  # Number of dice (strength)
        self.neighbors = []           # List of neighboring territory IDs
        self.highlighted = False      # Visual selection state
        self.features = {}            # Additional map features (resources, etc.)
        self.game = None              # Reference to the game object, set later
        
        # For faster point-in-polygon testing
        self.bbox = self._compute_bounding_box()
        
        # Precompute a convex hull for faster collision detection
        try:
            self.convex = self._compute_convex_hull()
        except Exception:
            # Fall back to original polygon if convex hull computation fails
            self.convex = None
    
    def _compute_bounding_box(self):
        """Compute the bounding box for this territory"""
        x_coords = [p[0] for p in self.polygon]
        y_coords = [p[1] for p in self.polygon]
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
    
    def _compute_convex_hull(self):
        """Compute a convex hull for faster collision detection"""
        if len(self.polygon) < 3:
            return self.polygon
            
        try:
            from scipy.spatial import ConvexHull
            points = np.array(self.polygon)
            hull = ConvexHull(points)
            return [tuple(points[i]) for i in hull.vertices]
        except Exception:
            return None
    
    def roll_dice(self):
        """Roll the dice and return the sum"""
        return sum(random.randint(1, 6) for _ in range(self.dice))
    
    def add_neighbor(self, territory_id):
        """Add a neighboring territory by ID"""
        if territory_id not in self.neighbors:
            self.neighbors.append(territory_id)
    
    def is_border_territory(self):
        """Check if this territory borders territories of other players/neutral"""
        if self.owner == -1:  # Neutral territories aren't border territories
            return False
            
        for neighbor_id in self.neighbors:
            if neighbor_id == -1:  # Edge of map
                continue
                
            # If any neighbor has different owner, this is a border territory
            if self.game.territories[neighbor_id].owner != self.owner:
                return True
                
        return False
        
    def point_in_territory(self, x, y):
        """Fast check if a point is inside this territory"""
        # Quick bounding box test first
        bbox = self.bbox
        if not (bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]):
            return False
            
        # Use convex hull test if available
        if self.convex is not None and len(self.convex) >= 3:
            return self._point_in_polygon(x, y, self.convex)
            
        # Fall back to original polygon if needed
        return self._point_in_polygon(x, y, self.polygon)
    
    def _point_in_polygon(self, x, y, polygon):
        """Ray casting algorithm for point-in-polygon test"""
        inside = False
        j = len(polygon) - 1
        
        for i in range(len(polygon)):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
                
            j = i
            
        return inside
    
    def render(self, renderer, cache):
        """Render this territory on the screen"""
        # Choose color based on owner
        color = NEUTRAL_COLOR if self.owner == -1 else PLAYER_COLORS[self.owner]
        
        # Use cached texture if available
        texture, min_x, min_y, width, height = cache.get_polygon_texture(self.polygon, color)
        
        # Create a destination rectangle
        dest_rect = sdl2.SDL_Rect(int(min_x), int(min_y), width, height)
        
        # Render the cached texture
        sdl2.SDL_RenderCopy(renderer, texture, None, dest_rect)
        
        # Draw the border if highlighted
        if self.highlighted:
            draw_polygon(renderer, self.polygon, HIGHLIGHT_COLOR)
            
        # Draw the dice using cached texture
        dice_color = NEUTRAL_COLOR if self.owner == -1 else PLAYER_COLORS[self.owner]
        dice_texture = cache.get_dice_texture(self.dice, dice_color)
        
        # Create a destination rectangle for the dice
        dice_size = 32
        dice_rect = sdl2.SDL_Rect(
            int(self.center[0] - dice_size // 2),
            int(self.center[1] - dice_size // 2),
            dice_size,
            dice_size
        )
        
        # Render the dice texture
        sdl2.SDL_RenderCopy(renderer, dice_texture, None, dice_rect)


class CFMapWars:
    """Main game class for CFMapWars"""
    
    def __init__(self, shapefile_path=None):
        """Initialize the game"""
        # Initialize SDL2
        sdl2.ext.init()
        self.window = sdl2.ext.Window(
            "CFMapWars", size=(WINDOW_WIDTH, WINDOW_HEIGHT)
        )
        self.window.show()
        self.ext_renderer = sdl2.ext.Renderer(self.window)
        self.renderer = self.ext_renderer.sdlrenderer
        
        # Set font to None to indicate we're not using text rendering
        self.font = None
        
        # Game state
        self.territories = {}
        self.num_players = 4
        self.current_player = 0
        self.selected_territory = None
        self.game_over = False
        self.winner = None
        self.turn_number = 1
        # self.human_player = 0  # Player 0 is human, others are AI
        
        # Create rendering cache
        self.render_cache = RenderCache(self.renderer)
        
        # Load map data - shapefile path is required
        if not shapefile_path or not os.path.exists(shapefile_path):
            raise ValueError(f"Shapefile not found: {shapefile_path}")
            
        self.load_map_data(shapefile_path)
        
        # Set the game reference in territories
        for territory in self.territories.values():
            territory.game = self
            
        # Create spatial index for faster territory lookups
        self.spatial_index = SpatialIndex(self.territories)
        
        # Precompute neighbor data
        self._precompute_neighbors()
        
        # Store player territory data for faster AI decisions
        self.player_territories = [[] for _ in range(self.num_players)]
        self.border_territories = [[] for _ in range(self.num_players)]
        
        # Initialize game
        self.initialize_players()
        
        # Precompute attack possibilities
        self.attack_possibilities = {}
        self._update_attack_possibilities()
    
    def _precompute_neighbors(self):
        """Precompute neighbor data to speed up AI decision making"""
        for territory in self.territories.values():
            territory.neighbor_territories = [
                self.territories[n_id] for n_id in territory.neighbors 
                if n_id in self.territories
            ]
    
    def _update_player_territories(self):
        """Update cached lists of player territories"""
        # Clear existing lists
        for i in range(self.num_players):
            self.player_territories[i].clear()
            self.border_territories[i].clear()
        
        # Rebuild lists
        for territory in self.territories.values():
            if territory.owner >= 0 and territory.owner < self.num_players:
                self.player_territories[territory.owner].append(territory)
                
                # Check if it's a border territory
                is_border = False
                for neighbor_id in territory.neighbors:
                    if neighbor_id in self.territories:
                        neighbor = self.territories[neighbor_id]
                        if neighbor.owner != territory.owner:
                            is_border = True
                            break
                            
                if is_border:
                    self.border_territories[territory.owner].append(territory)
    
    def _update_attack_possibilities(self):
        """Precompute potential attack moves"""
        self._update_player_territories()
        self.attack_possibilities.clear()
        
        # For each player
        for player in range(self.num_players):
            player_attacks = []
            
            # Check each territory owned by the player
            for territory in self.player_territories[player]:
                if territory.dice <= 1:
                    continue  # Need at least 2 dice to attack
                
                # Check neighbors for potential targets
                for neighbor_id in territory.neighbors:
                    if neighbor_id in self.territories:
                        neighbor = self.territories[neighbor_id]
                        
                        # Only consider attacks against other players or neutral
                        if neighbor.owner != player:
                            # Calculate attack strength
                            strength = territory.dice - neighbor.dice
                            player_attacks.append((territory, neighbor, strength))
            
            # Sort attacks by strength
            player_attacks.sort(key=lambda x: x[2], reverse=True)
            self.attack_possibilities[player] = player_attacks
    
    def load_map_data(self, shapefile_path):
        """Load real map data from shapefile"""
        # Open the shapefile
        ds = ogr.Open(shapefile_path)
        if ds is None:
            raise ValueError(f"Could not open shapefile {shapefile_path}")
        
        layer = ds.GetLayer(0)
        
        # Get the extent to map to screen coordinates
        x_min, x_max, y_min, y_max = layer.GetExtent()
        
        # Function to map geo coordinates to screen coordinates
        def map_to_screen(x, y):
            screen_x = ((x - x_min) / (x_max - x_min)) * (WINDOW_WIDTH - 100) + 50
            screen_y = WINDOW_HEIGHT - ((y - y_min) / (y_max - y_min)) * (WINDOW_HEIGHT - 100) - 50
            return screen_x, screen_y
        
        # Process each feature
        territory_bounds = {}  # Store bounding boxes for rtree
        
        for feature in layer:
            # Get geometry
            geom = feature.GetGeometryRef()
            
            # Handle multi-polygons (take the largest part)
            if geom.GetGeometryName() == "MULTIPOLYGON":
                largest_area = 0
                largest_poly = None
                
                for i in range(geom.GetGeometryCount()):
                    poly = geom.GetGeometryRef(i)
                    area = poly.GetArea()
                    if area > largest_area:
                        largest_area = area
                        largest_poly = poly
                
                geom = largest_poly
            
            # Get the polygon points
            ring = geom.GetGeometryRef(0)
            points = []
            
            for i in range(ring.GetPointCount()):
                x, y, _ = ring.GetPoint(i)
                screen_x, screen_y = map_to_screen(x, y)
                points.append((screen_x, screen_y))
            
            # Skip invalid polygons
            if len(points) < 3:
                continue
                
            # Get centroid
            centroid = geom.Centroid()
            center_x, center_y = map_to_screen(centroid.GetX(), centroid.GetY())
            
            # Get territory ID
            territory_id = feature.GetFID()
            
            # Try to get name from the shapefile attributes
            name = self._get_name_from_feature(feature) or f"Region {territory_id}"
            
            # Create the territory
            self.territories[territory_id] = Territory(
                territory_id, name, points, (center_x, center_y)
            )
            
            # Calculate bounding box for the territory
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            bbox = (min(x_coords), min(y_coords), max(x_coords), max(y_coords))
            territory_bounds[territory_id] = bbox
        
        print(f"Loaded {len(self.territories)} territories from shapefile")
        
        # If no territories were loaded, raise an error
        if len(self.territories) == 0:
            raise ValueError("No territories loaded from shapefile")
            
        # Identify neighbors using rtree
        self._identify_neighbors_with_rtree(territory_bounds)
    
    def _identify_neighbors_with_rtree(self, territory_bounds):
        """Identify neighboring territories using rtree spatial index"""
        from rtree import index
        
        # Create rtree index
        idx = index.Index()
        
        # Insert territories into the index
        for territory_id, bounds in territory_bounds.items():
            idx.insert(territory_id, bounds)
        
        # For each territory, find potential neighbors
        for territory_id, bounds in territory_bounds.items():
            territory = self.territories[territory_id]
            
            # Query index for overlapping and neighboring territories
            # Add a small buffer to catch adjacent territories
            buffer = 10  # pixels
            search_bounds = (
                bounds[0] - buffer,
                bounds[1] - buffer,
                bounds[2] + buffer,
                bounds[3] + buffer
            )
            
            for neighbor_id in idx.intersection(search_bounds):
                if neighbor_id == territory_id:
                    continue  # Skip self
                
                # Further verify using polygon checks
                if self._polygons_are_neighbors(
                    territory.polygon, 
                    self.territories[neighbor_id].polygon
                ):
                    territory.add_neighbor(neighbor_id)
    
    def _polygons_are_neighbors(self, poly1, poly2):
        """Check if two polygons are neighbors using an approximation"""
        # Find minimum distance between any point in poly1 and poly2
        min_distance = float('inf')
        
        # Sample some points from the perimeter for efficiency
        sample_rate = max(1, len(poly1) // 10)  # Sample about 10% of points
        samples1 = [poly1[i] for i in range(0, len(poly1), sample_rate)]
        
        sample_rate = max(1, len(poly2) // 10)
        samples2 = [poly2[i] for i in range(0, len(poly2), sample_rate)]
        
        # Calculate minimum distance
        for p1 in samples1:
            for p2 in samples2:
                distance = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
                min_distance = min(min_distance, distance)
                
                # Early exit if very close
                if min_distance < 30:  # 30 pixels threshold
                    return True
        
        # Consider them neighbors if they're close enough
        return min_distance < 50  # 50 pixels threshold
    
    # lol, bullshit: ... did not work ;)
    def _get_name_from_feature(self, feature):
        """Extract a name from the feature's attributes"""
        # Try common field names for regional names
        #common_name_fields = ["NAME", "NAME_1", "NUTS_NAME", "REGION", "LABEL", 
        #                      "NUTS_ID", "CNTR_CODE", "ID", "REGION_ID", "TERRITORY"]

        #
        #for field_name in common_name_fields:
        #    field_index = feature.GetFieldIndex(field_name)
        #    if field_index >= 0:
        #        return feature.GetField(field_index)

        field_idx = feature.GetFieldIndex('na') # <- it was 'na'; but granted one has to know that by inspecting the shape file ...
        return  feature.GetField(field_idx)
        
        # If no suitable field is found, generate a name
        return self._generate_region_name()
    
    def _generate_region_name(self):
        """Generate a historically-inspired region name based on NUTS-3 style"""
        prefixes = ["North", "South", "East", "West", "Upper", "Lower", "Central", "Old", "New"]
        roots = [
            "Saxony", "Bavaria", "Rhine", "Danube", "Elbe", "Loire", "Thames", "Tiber", 
            "Catalonia", "Andalusia", "Normandy", "Brittany", "Flanders", "Bohemia", "Moravia",
            "Swabia", "Burgundy", "Lombardy", "Tuscany", "Savoy", "Piedmont", "Gascony"
        ]
        suffixes = ["County", "Province", "Region", "District", "Land", "Valley", "Hills", "March"]
        
        # Different name patterns
        name_type = random.random()
        
        if name_type < 0.3:
            # Simple name
            return random.choice(roots)
        elif name_type < 0.6:
            # Prefix + root
            return f"{random.choice(prefixes)} {random.choice(roots)}"
        else:
            # Root + suffix
            return f"{random.choice(roots)} {random.choice(suffixes)}"
    
    def initialize_players(self):
        """Initialize player territories and dice"""
        # Get random territories for each player
        territory_ids = list(self.territories.keys())
        random.shuffle(territory_ids)
        
        # Calculate territories per player
        territories_per_player = min(5, len(territory_ids) // self.num_players)
        
        # Assign territories to players
        for player in range(self.num_players):
            for _ in range(territories_per_player):
                if territory_ids:
                    territory_id = territory_ids.pop()
                    territory = self.territories[territory_id]
                    territory.owner = player
                    territory.dice = random.randint(1, 3)
                    
        # Update player territory caches
        self._update_player_territories()
    
    def handle_events(self):
        """Process SDL2 events"""
        # Check for AI turns first
        if not self.game_over:
            self.ai_turn()
            return True
            
        # Process human events
        events = sdl2.ext.get_events()
        
        for event in events:
            if event.type == sdl2.SDL_QUIT:
                return False
            
            # Skip input handling if game is over
            if self.game_over:
                if event.type == sdl2.SDL_KEYDOWN:
                    if event.key.keysym.sym == sdl2.SDLK_SPACE:
                        # Reset game
                        self.__init__()
                continue
            
            # Handle mouse clicks
            if event.type == sdl2.SDL_MOUSEBUTTONDOWN:
                mouse_pos = (event.button.x, event.button.y)
                self._handle_mouse_click(mouse_pos)
            
            # Handle key presses
            if event.type == sdl2.SDL_KEYDOWN and event.key.keysym.sym == sdl2.SDLK_RETURN:
                    self.end_turn()
        
        return True
    
    def ai_turn(self):
        """Execute turn for AI player - optimized version"""
        print(f"AI Player {self.current_player} is taking its turn...")
        
        # Use cached player territories
        player_territories = self.player_territories[self.current_player]
        
        if not player_territories:
            # No territories, skip turn
            self.end_turn()
            return
        
        # Use precomputed attack possibilities
        potential_attacks = self.attack_possibilities.get(self.current_player, [])
        
        # Execute attacks
        attacks_made = 0
        for attacker, defender, _ in potential_attacks:
            if attacker.dice <= 1:
                continue  # Attacker may have been weakened by a previous attack
                
            if attacks_made >= 16:  # Limit to 3 attacks per turn
                break
                
            print(f"AI attacking from {attacker.name} ({attacker.dice} dice) to {defender.name} ({defender.dice} dice)")
            self.attack(attacker, defender)
            attacks_made += 1
            
            # Short delay to make AI turns visible
            sdl2.SDL_Delay(50)
            self.render()
        
        # End turn
        sdl2.SDL_Delay(50)  # Pause before ending turn
        self.end_turn()
    
    def _handle_mouse_click(self, pos):
        """Handle mouse click at the given position"""
        # Find the territory at this position using spatial index
        x, y = pos
        
        # Query spatial index for nearest territories
        nearest_ids = self.spatial_index.find_nearest((x, y), k=5)
        
        # Test if any of these territories contains the point
        clicked_territory = None
        for t_id in nearest_ids:
            territory = self.territories[t_id]
            if territory.point_in_territory(x, y):
                clicked_territory = territory
                break
        
        if clicked_territory:
            if self.selected_territory is None:
                # Try to select this territory if it belongs to current player
                if clicked_territory.owner == self.current_player and clicked_territory.dice > 1:
                    self.selected_territory = clicked_territory
                    clicked_territory.highlighted = True
            else:
                # Territory already selected - check for attack
                if clicked_territory.id in self.selected_territory.neighbors:
                    if clicked_territory.owner != self.current_player:
                        # Attack!
                        self.attack(self.selected_territory, clicked_territory)
                
                # Deselect territory
                self.selected_territory.highlighted = False
                self.selected_territory = None
    
    def attack(self, attacker, defender):
        """Process an attack from one territory to another"""
        # Ensure attacker has enough dice
        if attacker.dice <= 1:
            return
        
        # Roll dice for attack
        attack_roll = attacker.roll_dice()
        defense_roll = defender.roll_dice()
        
        print(f"Attack: {attacker.name} ({attack_roll}) -> {defender.name} ({defense_roll})")
        
        # Determine winner
        if attack_roll > defense_roll:
            # Attacker wins
            defender.owner = attacker.owner
            defender.dice = attacker.dice - 1
            attacker.dice = 1
            
            # Additional gameplay mechanic: Chance for bonus die on conquest
            if random.random() < 0.2:  # 20% chance
                if defender.dice < MAX_DICE:
                    defender.dice += 1
            
            # Update territory caches
            self._update_player_territories()
            
            # Update attack possibilities
            self._update_attack_possibilities()
            
            # Check if game is over
            self._check_game_over()
        else:
            # Defender wins
            attacker.dice = 1
            
            # Update attack possibilities since attacker's strength changed
            self._update_attack_possibilities()
    
    def end_turn(self):
        """End the current player's turn and process reinforcements"""
        # Calculate reinforcements
        self._distribute_reinforcements()
        
        # Move to next player
        self.current_player = (self.current_player + 1) % self.num_players
        
        # Skip players who have been eliminated
        while not self.player_territories[self.current_player]:
            self.current_player = (self.current_player + 1) % self.num_players
            
            # If we've gone full circle, check game over
            if self.current_player == 0:
                active_players = sum(1 for territories in self.player_territories if territories)
                if active_players <= 1:
                    self._check_game_over()
                    return
        
        # Increment turn number every time we get back to player 0
        if self.current_player == 0:
            self.turn_number += 1
            
        # Update attack possibilities for the new player
        self._update_attack_possibilities()
    
    def _distribute_reinforcements(self):
        """Distribute reinforcements to the current player's territories"""
        # Use cached player territories
        player_territories = self.player_territories[self.current_player]
        territory_count = len(player_territories)
        
        if territory_count == 0:
            return
        
        # Base reinforcements + 1 per 3 territories
        reinforcements = BASE_REINFORCEMENTS + (territory_count // 3)
        
        # Use cached border territories
        border_territories = self.border_territories[self.current_player]
        
        # If we have border territories, reinforce those first
        if border_territories:
            # Sort by dice count (reinforce weaker territories first)
            border_territories.sort(key=lambda t: t.dice)
            
            # Use 70% of reinforcements on borders
            border_reinforcements = int(reinforcements * 0.7)
            remaining_reinforcements = reinforcements - border_reinforcements
            
            # Reinforce border territories
            for territory in border_territories:
                if border_reinforcements <= 0:
                    break
                    
                if territory.dice < MAX_DICE:
                    add_dice = min(MAX_DICE - territory.dice, border_reinforcements)
                    territory.dice += add_dice
                    border_reinforcements -= add_dice
            
            # Use remaining reinforcements on interior territories
            interior_territories = [t for t in player_territories if t not in border_territories]
            interior_territories.sort(key=lambda t: t.dice)
            
            for territory in interior_territories:
                if remaining_reinforcements <= 0:
                    break
                    
                if territory.dice < MAX_DICE:
                    add_dice = min(MAX_DICE - territory.dice, remaining_reinforcements)
                    territory.dice += add_dice
                    remaining_reinforcements -= add_dice
        else:
            # No border territories, reinforce all territories evenly
            # Sort territories by dice count (reinforce weaker territories first)
            player_territories.sort(key=lambda t: t.dice)
            
            # Distribute reinforcements
            remaining = reinforcements
            while remaining > 0 and any(t.dice < MAX_DICE for t in player_territories):
                for territory in player_territories:
                    if remaining <= 0:
                        break
                        
                    if territory.dice < MAX_DICE:
                        territory.dice += 1
                        remaining -= 1
        
        # Update attack possibilities since territory strengths changed
        self._update_attack_possibilities()
    
    def _check_game_over(self):
        """Check if the game is over (one player owns all territories)"""
        active_players = set()
        for territory in self.territories.values():
            if territory.owner >= 0:  # Not neutral
                active_players.add(territory.owner)
        
        if len(active_players) == 1:
            self.game_over = True
            self.winner = next(iter(active_players))
            print(f"Game over! Player {self.winner + 1} wins!")
    
    def render(self):
        """Render the game"""
        # Clear screen with background color
        sdl2.SDL_SetRenderDrawColor(
            self.renderer, 
            BG_COLOR[0], BG_COLOR[1], BG_COLOR[2], BG_COLOR[3]
        )
        sdl2.SDL_RenderClear(self.renderer)
        
        # Draw territories
        for territory in self.territories.values():
            territory.render(self.renderer, self.render_cache)
        
        # Draw UI - using shapes instead of text
        if not self.game_over:
            # Draw current player indicator (colored rectangle at top)
            player_color = PLAYER_COLORS[self.current_player]
            indicator_rect = sdl2.SDL_Rect(10, 10, 100, 30)
            
            # Draw the rectangle
            sdl2.SDL_SetRenderDrawColor(
                self.renderer,
                player_color[0], player_color[1], player_color[2], player_color[3]
            )
            sdl2.SDL_RenderFillRect(self.renderer, indicator_rect)
            
            # Draw border
            sdl2.SDL_SetRenderDrawColor(
                self.renderer,
                BORDER_COLOR[0], BORDER_COLOR[1], BORDER_COLOR[2], BORDER_COLOR[3]
            )
            sdl2.SDL_RenderDrawRect(self.renderer, indicator_rect)
            
            # Draw AI indicator if current player is AI
            ai_indicator_rect = sdl2.SDL_Rect(120, 10, 60, 30)
            sdl2.SDL_SetRenderDrawColor(
                self.renderer,
                200, 200, 200, 255  # Light gray
            )
            sdl2.SDL_RenderFillRect(self.renderer, ai_indicator_rect)
            sdl2.SDL_SetRenderDrawColor(
                self.renderer,
                BORDER_COLOR[0], BORDER_COLOR[1], BORDER_COLOR[2], BORDER_COLOR[3]
            )
            sdl2.SDL_RenderDrawRect(self.renderer, ai_indicator_rect)
            
            # Draw turn number indicator (small circles)
            for i in range(min(self.turn_number, 10)):  # Only show up to 10 turn indicators
                # Draw a small circle for each turn
                x = 200 + i * 20
                y = 25
                radius = 5
                
                # Fill circle
                # draw_filled_circle(self.renderer, x, y, radius, BORDER_COLOR)
        else:
            # Draw game over indicator (large rectangle in center)
            winner_color = PLAYER_COLORS[self.winner]
            indicator_rect = sdl2.SDL_Rect(
                WINDOW_WIDTH // 2 - 150, 
                WINDOW_HEIGHT // 2 - 50,
                300, 100
            )
            
            # Draw the rectangle
            sdl2.SDL_SetRenderDrawColor(
                self.renderer,
                winner_color[0], winner_color[1], winner_color[2], winner_color[3]
            )
            sdl2.SDL_RenderFillRect(self.renderer, indicator_rect)
            
            # Draw border
            sdl2.SDL_SetRenderDrawColor(
                self.renderer,
                BORDER_COLOR[0], BORDER_COLOR[1], BORDER_COLOR[2], BORDER_COLOR[3]
            )
            sdl2.SDL_RenderDrawRect(self.renderer, indicator_rect)
        
        # Present the rendered frame
        sdl2.SDL_RenderPresent(self.renderer)
    
    def run(self):
        """Main game loop"""
        running = True
        frame_count = 0
        last_time = sdl2.SDL_GetTicks()
        
        while running:
            # Handle events
            running = self.handle_events()
            
            # Render
            self.render()
            
            # Calculate and display FPS every 10 frames
            frame_count += 1
            if frame_count >= 10:
                current_time = sdl2.SDL_GetTicks()
                elapsed = current_time - last_time
                fps = frame_count / (elapsed / 1000.0) if elapsed > 0 else 0
                
                # Set window title with FPS
                self.window.title = f"CFMapWars - FPS: {fps:.1f}"
                
                # Reset counters
                frame_count = 0
                last_time = current_time
            
            # Delay to control framerate
            sdl2.SDL_Delay(16)  # ~60 FPS
        
        # Clean up
        self.render_cache.cleanup()
        sdl2.ext.quit()


def main():
    """Main entry point for the game"""
    # Parse command line arguments for shapefile path
    if len(sys.argv) < 2:
        print("Usage: python cfmapwars.py <shapefile_path>")
        sys.exit(1)
        
    shapefile_path = sys.argv[1]
    
    try:
        # Create and run the game
        game = CFMapWars(shapefile_path)
        game.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

