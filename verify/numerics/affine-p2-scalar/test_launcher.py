"""Initial one-candidate launcher bindings; pure tests with no Eqiora import."""
from fractions import Fraction
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

_spec=importlib.util.spec_from_file_location('_p2_initial_launcher',Path(__file__).with_name('check.py'))
launcher=importlib.util.module_from_spec(_spec);_spec.loader.exec_module(launcher)


class LauncherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.digest=launcher.sha256((launcher.HERE/'reference'/'data'/'manifest.json').read_bytes())
        cls.packet,cls.gate=launcher.reference(cls.digest,2)
    def candidate(self,packet=None):
        p=self.packet if packet is None else packet
        return {'schema':1,'dimension':p['dimension'],'N':2,'profile':'patch','gauss_points':3,'records':[
            {'coordinates_m':[str(float(Fraction(x))) for x in coordinate],
             'value_binary64_hex':float(Fraction(value)).hex()}
            for coordinate,value in zip(p['nodes_m'],p['exact_coefficients'])]}
    def test_exact_reference_vector_uses_frozen_gate_without_native_actions(self):
        result=launcher.compare_patch(self.candidate(),self.packet,type(self).gate)
        self.assertTrue(result['accepted']);self.assertFalse(result['native_constrained_action_witness_present'])
    def test_3d_frozen_packet_and_dimension_metadata_binding(self):
        packet,gate=launcher.reference(self.digest,3)
        candidate=self.candidate(packet)
        self.assertEqual(len(candidate['records']),125)
        self.assertTrue(launcher.compare_patch(candidate,packet,gate)['accepted'])
        for key,value in [('dimension',2),('N',4),('profile','refinement'),('gauss_points',4)]:
            altered={**candidate,key:value}
            with self.assertRaises(ValueError):launcher.compare_patch(altered,packet,gate)
        with self.assertRaises(ValueError):launcher.compare_patch(candidate,self.packet,type(self).gate)
        with self.assertRaises(ValueError):launcher.reference(self.digest,4)
    def test_complete_coordinate_and_nonfinite_falsifiers(self):
        for mutant in ('missing','duplicate','nonfinite','negated'):
            candidate=self.candidate();records=candidate['records']
            if mutant=='missing':records.pop()
            elif mutant=='duplicate':records[-1]=dict(records[0])
            elif mutant=='nonfinite':records[0]['value_binary64_hex']='nan'
            else:
                for row in records:row['value_binary64_hex']=(-float.fromhex(row['value_binary64_hex'])).hex()
            with self.assertRaises(ValueError):launcher.compare_patch(candidate,self.packet,type(self).gate)
    def test_reference_digest_is_required(self):
        with self.assertRaises(ValueError):launcher.reference('0'*64,2)
    def test_native_receipt_requires_successful_reviewed_source_and_digest(self):
        good={'source_revision':'a'*40,'actual_terminal_exit_code':0,'installed_module':{'native_sha256':'b'*64}}
        launcher.native_receipt_matches(good,'a'*40,'b'*64)
        for source,digest,code in [('c'*40,'b'*64,0),('a'*40,'c'*64,0),('a'*40,'b'*64,1)]:
            altered={**good,'actual_terminal_exit_code':code}
            with self.assertRaises(ValueError):launcher.native_receipt_matches(altered,source,digest)
    def test_parser_allows_only_two_dimensions_and_no_level_profile_or_benchmark_switch(self):
        options={option for action in launcher.parser()._actions for option in action.option_strings}
        self.assertIn('--dimension',options)
        self.assertEqual(next(a for a in launcher.parser()._actions if '--dimension' in a.option_strings).choices,(2,3))
        self.assertFalse(options & {'--cells','--profile','--samples','--warmups'})

if __name__=='__main__':unittest.main()
