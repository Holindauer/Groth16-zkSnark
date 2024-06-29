from QAP import QAP
from trusted_setup import TrustedSetup
from prover import Prover
from verifier import Verifier

if __name__ == "__main__":
    
    # QAP for 40 = x^3 - 4x^2 + 6x + y^2
    qap = QAP()

    # convert R1CS to QAP
    U_poly, V_poly, W_poly, t = qap.R1CS_to_QAP()

    # get galois field 
    GF = qap.GF
    curve_order = qap.curve_order

    # trusted setup
    ts = TrustedSetup(U_poly, V_poly, W_poly, t, GF, curve_order)
    setup = ts.setup(degree=4)
    
    # init prover w/ QAPs and setup
    prover = Prover(U_poly, V_poly, W_poly, setup)
    verifier = Verifier(setup)

    # generate a valid proof
    print("generating a valid proof...")
    valid_proof, valid_claim = prover.genProof(3, 2)
    A, B, C = valid_proof
    result = verifier.verifyProof(A, B, C, valid_claim)
    assert result, "valid proof failed verification"

    # generate an invalid proof
    invalid_proof, invalid_claim = prover.genProof(3, 3)
    A, B, C = invalid_proof
    result = verifier.verifyProof(A, B, C, invalid_claim)
    assert not result, "invalid proof passed verification"

    print("all tests passed")