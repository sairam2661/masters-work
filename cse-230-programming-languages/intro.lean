open Nat

def add (n m : Nat) : Nat :=
  match n with
  | zero => m
  | succ n' => succ (add n' m)

#eval add 4 5


def add_tr(xs ys : Nat) : Nat :=
  match xs with
  | zero => ys
  | succ xs' => add_tr xs' (succ ys)

def fact(n : Nat) : Nat :=
  match n with
  | zero => 1
  | succ n' => Nat.mul (succ n') (fact n')

#eval add_tr 4 2

def sum_list (xs : List Nat) : Nat :=
  match xs with
  | [] => 0
  | x ::xs' => x + sum_list xs'

def sum_list_tr (xs : List Nat) (acc : Nat): Nat :=
  match xs with
  | [] => acc
  | x :: ys => sum_list_tr ys (acc + x)


-- tail_ih✝ : sum_list' tail✝ = sum_list tail✝
-- ⊢ sum_list' (head✝ :: tail✝) = sum_list (head✝ :: tail✝)





  -- intros e
  -- simp [eval', generalized_ih]

/- ***********************************************************************************
   Problem 5 (NK: exercise 2.10)
   *********************************************************************************** -/


/- ***********************************************************************************
   Problem 5 (NK: exercise 2.10)
   *********************************************************************************** -/

-- HINT: You may find this theorem useful in your proof
theorem mul_shuffle : ∀ (a b c : Nat), a * (b * c) = b * (a * c) := by
  intros a b c
  calc
    a * (b * c) = (a * b) * c := by simp [Nat.mul_assoc]
    _           = (b * a) * c := by simp [Nat.mul_comm]
    _           = b * (a * c) := by simp [Nat.mul_assoc]

-- An `Exp` datatype to represent polynomials over a variable `x`
-- e ::= n | x | e + e | e * e

inductive Exp where
 | Var : Exp
 | Con : Nat -> Exp
 | Add : Exp -> Exp -> Exp
 | Mul : Exp -> Exp -> Exp
 deriving Repr

open Exp

-- `poly0` is a representation of `x + 10`
def poly0 : Exp := Add (Var) (Con 10)

-- `poly1` is a representation of `2x^2`
def poly1 : Exp := Mul (Con 2) (Mul Var Var)

-- `poly2` is a representation of `2x^2 + x + 10`
def poly2 : Exp := Add poly1 poly0

-- (a) Complete the definition of a function `eval` such that `eval e x` evaluates `e` at the value `x`;
-- when you are done, `eval_test` should be automatically checked.

def eval (e: Exp) (x: Nat) : Nat :=
  match e with
  | Exp.Var => x
  | Exp.Con n => n
  | Exp.Add e1 e2 => eval e1 x + eval e2 x
  | Exp.Mul e1 e2 => eval e1 x * eval e2 x

theorem eval_test : eval poly2 5 = 65 := rfl

-- A "compact" representation of polynomials as a list of coefficients, starting with the constant
-- For example, `[4, 2, 1, 3]` represents the polynomial `4 + 2.x + 1.x^2 + 3.x^3`, and
-- [10,1,2] represents the polynomial `10 + 1.x + 2.x^2` (i.e. `poly2`)

abbrev Poly := List Nat

-- (b) Complete the implementation of `evalp` so that `evalp_test` succeeds automatically
def evalp (p: Poly) (x: Nat) : Nat :=
 match p with
 | [] => 0
 | y::ys => evalp ys x * x + y

theorem evalp_test : evalp [10, 1, 2] 5 = eval poly2 5 := rfl

-- (c) Complete the implementation of `coeffs` so that `coeffs_test` succeeds automatically
-- HINT: you may need helper functions `addp` and `mulp` which *add* and *multiply* two `Poly`s
def addp (p1 p2 : Poly) : Poly :=
  match p1, p2 with
  | [], p2 => p2
  | p1, [] => p1
  | x::xs, y::ys => (x + y) :: addp xs ys

-- #eval addp [1,2,3] [4,5,6,7]

def mulp (p1 p2 : Poly) : Poly :=
  match p1 with
  | [] => []
  | x::xs => addp (List.map (λ (y: Nat) => x * y) p2 ) (0::mulp xs p2)

-- #eval mulp [1,2,3] [4,0,1]

def coeffs (e: Exp) : Poly :=
  match e with
  | Exp.Con n => [n]
  | Exp.Var => [0,1]
  | Exp.Add e1 e2 => addp (coeffs e1) (coeffs e2)
  | Exp.Mul e1 e2 => mulp (coeffs e1) (coeffs e2)

theorem coeffs_test : coeffs poly2 = [10, 1, 2] := by rfl


-- (d) Complete the proof of `eval_poly`; HINT: you will likely
-- require helper lemmas e.g. about the helper functions `addp` and `mulp`...

theorem addp_helper : ∀ (p1 p2 : Poly) (x : Nat),
  evalp (addp p1 p2) x = evalp p1 x + evalp p2 x := by
  intros p1 p2 x
  induction p1 generalizing p2
  . case nil =>
    simp [addp, evalp]
  . case cons head tail ih =>
    cases p2
    . simp [addp, evalp]
    . simp [addp, evalp, ih, Nat.mul_add, Nat.add_mul, Nat.add_assoc, Nat.add_comm, Nat.add_left_comm]

theorem map_evalp : ∀ (k: Nat) (p: Poly) (x: Nat),
  evalp (List.map (fun y => k * y) p) x = k * evalp p x := by
  intros k p x
  induction p
  . case nil =>  simp [List.map, evalp]
  . case cons y ys ih =>
    simp [evalp, ih, Nat.mul_add, Nat.mul_assoc]

theorem mulp_helper : ∀ (p1 p2 : Poly) (x : Nat),
 evalp (mulp p1 p2) x = evalp p1 x * evalp p2 x := by
intros p1 p2 x
induction p1
. case nil =>
  simp [mulp, evalp]
. case cons y ys ih => simp [map_evalp, addp_helper, evalp, ih, Nat.add_assoc, Nat.add_left_comm, mul_shuffle, Nat.left_distrib, Nat.mul_comm]
  done

theorem eval_poly : ∀ (e:Exp) (x:Nat), evalp (coeffs e) x = eval e x := by
  intros e x
  induction e
  . case Var => simp [coeffs, eval, evalp]
  . case Con n => simp [coeffs, eval, evalp]
  . case Add e1 e2 e1_ih e2_ih => simp [addp, coeffs, eval, addp_helper, e1_ih, e2_ih]
  . case Mul e1 e2 e1_ih e2_ih => simp [mulp, coeffs, eval, mulp_helper, e1_ih, e2_ih]
  done

evalp (addp (List.map (fun y_1 => y * y_1) tail✝) (mulp ys (head✝ :: tail✝))) x * x + y * head✝ =
(evalp ys x * x + y) * (evalp tail✝ x * x + head✝)
