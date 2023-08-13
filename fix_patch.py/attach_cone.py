#! /usr/bin/env python
import igl
import numpy as np
import sys
from pathlib import Path


def printHelpExit():
    print("Invalid command line arguments specified!\n\n")
    print("inFile and outFIle are required. outFaces, upsample and offset default to 8000, 2 and 0,-10,2 respectively")
    print("USAGE: hole_fixer [options]\n\n")
    print("OPTIONS: \n")
    print("  -in\t\t\ttarget mesh file in .ply-format, with a hole\n")
    print("  -out\t\t\toutput mesh file in .ply-format\n")
    print("  -outfaces\t\tHow many faces to decimate the mesh to\n")
    print("  -upsample\t\tHow much upsampling to use when creating the patch\n")
    print("  -offset\t\tHow much offset to use when shifting the patch center\n")
    exit()


class ArgvParser():
    @staticmethod
    def findToken(param, argv):
        """_summary_

        Args:
            param (_type_): _description_
            argv (_type_): _description_

        Returns:
            _type_: _description_
        """
        token = None
        if param in argv:
            idx = argv.index(param)
            try:
                token = argv[idx + 1]
            except IndexError:
                pass
        return token

    @staticmethod
    def parseStringParam(param, argv):
        return ArgvParser.findToken(param, argv)

    @staticmethod
    def parseIntParam(param, argv):
        token = ArgvParser.findToken(param, argv)
        if token:
            try:
                token = int(token)
            except ValueError:
                print(f"Unknown value for {param}\n")
                printHelpExit()
        return token

    @staticmethod
    def parseCoordParam(param, argv):
        token = ArgvParser.findToken(param, argv)
        if token:
            try:
                x, y, z = map(float, token.split(','))
                print(x,y,z)
                return([x, y, z])
            except ValueError:
                print(f"Unknown value for {param}\n")
                printHelpExit()
        else:
            return None

    @staticmethod
    def parse_arguments(argv):
        inFIle = ArgvParser.parseStringParam("-in", argv)
        outFIle = ArgvParser.parseStringParam("-out", argv)
        outFacesN = ArgvParser.parseIntParam("-outfaces", argv)
        upsampleN = ArgvParser.parseIntParam("-upsample", argv)
        offset = ArgvParser.parseCoordParam("-offset", argv)
        return inFIle, outFIle, outFacesN, upsampleN, offset 


def patch_tooth_with_cone(inFile, outFile, outFacesN=8000, upsampleN=2, DEBUG=False): # , offset=[0, -10, 2]
    """_summary_

    Args:
        inFile (_type_): _description_
        outFile (_type_): _description_
        outFacesN (int, optional): _description_. Defaults to 8000.
        upsampleN (int, optional): _description_. Defaults to 2.
        DEBUG (bool, optional): _description_. Defaults to False.

    Returns:
        _type_: _description_
    """
    vertex, face = igl.read_triangle_mesh(inFile)
    loop = igl.boundary_loop(face)
    vertex, face = igl.upsample(vertex, face, upsampleN)
    bcenter = (1.0 / len(loop)) * np.sum(vertex[loop], axis=0)
    # New dynamic offset
    x_dist = np.max(vertex[:, 0]) - np.min(vertex[:, 0])
    y_dist = np.max(vertex[:, 1]) - np.min(vertex[:, 1])
    z_dist = np.max(vertex[:, 2]) - np.min(vertex[:, 2])
    offset = np.array([0, -(np.max([x_dist, y_dist, z_dist])), 0])
    bcenter += offset

    patchV = vertex[loop]
    patchV = np.vstack([patchV, bcenter])
    patchF = np.zeros((loop.size, 3), dtype=np.int32)
    patchF[:, 0] = (np.arange(loop.size) + 1) % loop.size # np.arange(loop.size)
    patchF[:, 1] = np.arange(loop.size) # np.roll(patchF[:, 0], -1)
    patchF[:, 2] = loop.size
    patchV, patchF = igl.upsample(patchV, patchF, upsampleN)
    if DEBUG:
        igl.write_triangle_mesh("patch.ply", patchV, patchF)

    # TODO make a function as FuseMesh
    index = 0  # Vertex index counter for the fused mesh
    fusedV, fusedF = [], []
    # Add the upsampled patch to the fused mesh
    for i in range(patchV.shape[0]):
        fusedV.append([patchV[i, 0], patchV[i, 1], patchV[i, 2]])
        index += 1
    for i in range(patchF.shape[0]):
        fusedF.append([patchF[i, 0], patchF[i, 1], patchF[i, 2]])
    fusedV = np.array(fusedV)
    fusedF = np.array(fusedF)
    # Fuse the patch and the original mesh together
    originalToFusedMap = {}
    for itri in range(face.shape[0]):
        triIndices = np.zeros(3, dtype=int)
        for iv in range(3):
            triIndex = face[itri, iv]
            ret = None
            if triIndex not in originalToFusedMap:
                foundMatch = -1
                # The vertices at the boundary are the same for both the patch and original mesh.
                # We ensure that these vertices are not duplicated.
                # This is also how we ensure that the two meshes are fused together.
                for jj in range(patchV.shape[0]):
                    u = np.array(fusedV[jj])
                    v = vertex[triIndex]
                    if np.linalg.norm(u - v) < 0.00001:
                        foundMatch = jj
                        break
                if foundMatch != -1:
                    originalToFusedMap[triIndex] = foundMatch
                    ret = foundMatch
                else:
                    fusedV = np.vstack([fusedV, [vertex[triIndex, 0], vertex[triIndex, 1], vertex[triIndex, 2]]])
                    originalToFusedMap[triIndex] = index
                    ret = index
                    index += 1
            else:
                ret = originalToFusedMap[triIndex]
            triIndices[iv] = ret
        fusedF = np.vstack([fusedF, triIndices])
    # fusedV = np.array(fusedV)
    # fusedF = np.array(fusedF)
    if DEBUG:
        igl.write_triangle_mesh("fused.ply", fusedV, fusedF)
    
    # Convert fusedV to fairedV
    fairedV = np.zeros((fusedV.shape[0], 3))
    for vindex in range(fusedV.shape[0]):
        r = fusedV[vindex]
        fairedV[vindex, 0] = r[0]
        fairedV[vindex, 1] = r[1]
        fairedV[vindex, 2] = r[2]
    # Convert fusedF to fairedF
    fairedF = np.zeros((fusedF.shape[0], 3), dtype=int)
    for findex in range(fusedF.shape[0]):
        r = fusedF[findex]
        fairedF[findex, 0] = r[0]
        fairedF[findex, 1] = r[1]
        fairedF[findex, 2] = r[2]
    b = np.zeros(fairedV.shape[0] - patchV.shape[0], dtype=int)
    bc = np.zeros((fairedV.shape[0] - patchV.shape[0], 3))
    for i in range(patchV.shape[0], fairedV.shape[0]):
        jj = i - patchV.shape[0]
        b[jj] = i
        bc[jj, 0] = fairedV[i, 0]
        bc[jj, 1] = fairedV[i, 1]
        bc[jj, 2] = fairedV[i, 2]
    k = 2
    Z = igl.harmonic_weights(fairedV, fairedF, b, bc, k)
    fairedV = Z
    if DEBUG:
        igl.write_triangle_mesh("faired.ply", fairedV, fairedF)

    res, finalV, finalF, temp0, temp1 = igl.decimate(fairedV, fairedF, outFacesN)
    try:
        igl.write_triangle_mesh(outFile, finalV, finalF)
    except ValueError:
        pass
    outFile, inFile = Path(outFile), Path(inFile)
    print(outFile.parent/f"faired-{inFile.stem}.ply")
    igl.write_triangle_mesh(str(outFile.parent/f"faired-{inFile.stem}.ply"), fairedV, fairedF)
    return True



def main():
    inFile, outFile, outFacesN, upsampleN, offset = ArgvParser.parse_arguments(sys.argv)
    if inFile == None or outFile == None:
        printHelpExit()
    outFacesN = outFacesN if outFacesN else 8000
    upsampleN = upsampleN if upsampleN else 2
    offset = offset if offset else [0, -10, 2]
    # print(inFile, outFile, outFacesN, upsampleN, offset)
    ret = patch_tooth_with_cone(inFile, outFile, outFacesN, upsampleN, offset)
    print(ret)

if __name__ == "__main__":
    # main()
    inFile = "data/fixed/ALEX\ PLOTNER-mandibular_coloured_With_Normal.ply/purple.ply"
    outFile = "output.ply"
    patch_tooth_with_cone(inFile, outFile)