from molSimplify.Classes.mol3D import mol3D
from molSimplify.Scripts.generator import startgen_pythonic
from typing import cast

cmd_dict = {
    "-core": "pd",
    "-geometry": "sqp",
    "-coord": "4",
    "-lig": "pph3,bromide,c1ccccc1[CH-]C(c2ccccc2)",
    "-ligocc": "2,1,1",
    "-spin": "1",
    "-oxstate": "2",
    "-smicat": "[6]",
    "-name": "test",
}
stargen_thing = startgen_pythonic(input_dict=cmd_dict)
if stargen_thing is None:
    raise Exception("should not happen (probably)")
strfiles, emsg, result = stargen_thing

molecule = cast(mol3D, result.mol)

molecule.writexyz("test.xyz")
