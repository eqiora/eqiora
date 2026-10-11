# org.example.closed_circuit

An example application package that depends only on the public interface of
`Eqiora.Electrical.Circuits`. It instantiates the composed `ParallelDc`
component and sets its three scalar parameters.

It exists to test that a component can be reused through an intermediate
package offline, without reaching through to `Eqiora.Electrical.Basic`.
