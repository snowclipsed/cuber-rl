import numpy as np
import re

class CubeState:
    FACES = ['U', 'L', 'F', 'R', 'B', 'D']
    COLORS = {'W': 0, 'R': 1, 'B': 2, 'O': 3, 'G': 4, 'Y': 5}
    COLOR_CHARS = 'WRBOGY'
    
    def __init__(self, state_string=None):
        if state_string:
            self.state = self._parse(state_string)
        else:
            self.state = np.array([[[i]*3]*3 for i in range(6)])
    
    def _parse(self, s):
        state = np.zeros((6, 3, 3), dtype=int)
        pattern = r'(?:TOP\(U\)|LEFT\(L\)|FRONT\(F\)|RIGHT\(R\)|BACK\(B\)|BOTTOM\(D\)):\s*([WRBOGY]{3})/([WRBOGY]{3})/([WRBOGY]{3})'
        matches = re.findall(pattern, s)
        face_map = {'TOP(U)': 0, 'LEFT(L)': 1, 'FRONT(F)': 2, 'RIGHT(R)': 3, 'BACK(B)': 4, 'BOTTOM(D)': 5}
        
        for match_idx, (r1, r2, r3) in enumerate(matches):
            face_idx = list(face_map.values())[match_idx]
            for i, row in enumerate([r1, r2, r3]):
                for j, color in enumerate(row):
                    state[face_idx, i, j] = self.COLORS[color]
        return state
    
    def to_string(self):
        face_names = ['TOP(U)', 'LEFT(L)', 'FRONT(F)', 'RIGHT(R)', 'BACK(B)', 'BOTTOM(D)']
        lines = []
        for face_idx, name in enumerate(face_names):
            rows = ['/'.join(self.COLOR_CHARS[self.state[face_idx, i, j]] for j in range(3)) for i in range(3)]
            lines.append(f"{name}: {'/'.join(rows)}")
        return '\n'.join(lines)
    
    def to_magiccube(self):
        # MagicCube uses different indexing: convert our state to their format
        # Our order: U,L,F,R,B,D -> MagicCube typically uses same
        return self.state.copy()
    
    @classmethod
    def from_magiccube(cls, cube_array):
        obj = cls()
        obj.state = cube_array.copy()
        return obj
    
    def is_solved(self):
        return all(np.all(self.state[i] == self.state[i, 0, 0]) for i in range(6))
    
    def __eq__(self, other):
        return np.array_equal(self.state, other.state)
    
    def get_face(self, face):
        face_idx = self.FACES.index(face) if face in self.FACES else face
        return self.state[face_idx]