from ._base import *
    
class AusSync(parser_base):
    
    ALLOWED_EXTENSIONS = ['.asc'] #ascii files exported from the binary .mda files generated.
    
    @overrides.override
    def file_parser(file: TextIOWrapper) -> tuple[NDArray, list[str], dict[str, Any]]:
        """Reads Australian Synchrotron .asc files.

        Parameters
        ----------
        file : TextIOWrapper
            TextIOWrapper of the datafile (i.e. open('file.asc', 'r'))

        Returns
        -------
        tuple[NDArray, list[str], dict[str, Any]]
            Returns a set of data as a numpy array, 
            labels as a list of strings,
            and parameters as a dictionary.

        Raises
        ------
        ValueError
            If the file is not a valid .asc file.
        """
        
        # Init vars
        data = []
        labels = []
        params = []
        
        # Check file type
        if not file.name.endswith('.asc'):
            raise ValueError(f"File {file.name} is not a valid .asc file.")
        
        ### Read file
        # Check header is correct
        assert file.readline() == "## mda2ascii 0.3.2 generated output\n"
        
        #skip 2 lines
        [file.readline() for i in range(2)]
        
        # 1 Initial Parameters including MDA File v, Scan Num, Ovrl scn dim, scn size 
        for i in range(4):
            line = file.readline()[2:].strip().split(' = ', 1)
            if len(line) == 2:
                params[line[0]] = line[1]

        #skip 2 lines
        [file.readline() for i in range(2)]
        # Check PV header is correct
        assert file.readline() == "# Extra PV: name, descr, values (, unit)\n"
        #skip 1 line
        file.readline()

        # 2 Extra Parameter/Values
        line = file.readline()
        while line != '\n':
            # Discard first 11 characters and linenumber '# Extra PV 1:'.
            line = line.strip().split(":", 1)[1:]
            # Separate the name, description, values and units.
            line = line.split(',')
            # Trim values and append to params in a tuple.
            params[line[0].strip()] = [line[i].strip() for i in range(1,len(line))]
            
        # skip 1 line
        file.readline()
                        
        # 3 Scan Header:
        assert file.readline() == "# 1-D Scan"
        for i in range(3):
            line = file.readline()[2:].strip().split(' = ', 1)
            if len(line) == 2:
                params[line[0]] = line[1]
            
        #skip 1 line
        file.readline()
        
        
        # 
            
        return data, labels, params
    
    # class mda:
#     DTYPES = {
#         -1: 'complex float32',
#         -2: 'byte',
#         -3: 'float32',
#         -4: 'int16',
#         -5: 'int32',
#         -6: 'uint16',
#         -7: 'double',
#         -8: 'uint32'
#     }
    
#     @staticmethod
#     def read_mda(filename : str):
#         """Read .mda file according to https://mountainlab.readthedocs.io/en/latest/mda_file_format.html"""
        
#         with open(filename, 'rb') as f:
#             # Access header information
#             dint = int(f.read(4))
#             dtype = mda.DTYPES[dint]
#             dtype_bytes = int(f.read(4)) #redundant from DTYPES, but useful
#             dimensions = int(f.read(4)) #between 1 and 50
#             dimension_size = [] # size of each dimension
#             for i in range(dimensions):
#                 dimension_size.append(int(f.read(4)))
            
#             # Now raw data!
            
            

#         return